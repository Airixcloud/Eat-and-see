"""
Trash classifier: ResNet50 transfer learning with a correction queue.

Usage:
    python train.py                    # shows a menu: train on new data only, or full retrain
    python train.py --corrections-only # skip the menu: train on the new (corrected) data only
    python train.py --full             # skip the menu: full retrain from scratch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 64
SEED = 42

# Upper bounds; early stopping usually ends training sooner
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 8
EPOCHS_CORRECTION = 3

LR_INITIAL = 1e-3
LR_FINE_TUNE = 1e-5
LR_CORRECTION = 1e-5

UNFREEZE_LAYERS = 30        # last N ResNet layers to fine-tune (BatchNorm stays frozen)
EARLY_STOP_PATIENCE = 2
USE_CLASS_WEIGHTS = True

# Correction training
CORRECTION_REPEAT = 5       # each corrected image appears this many times
REPLAY_PER_CORRECTION = 10  # old training images mixed in per correction (prevents forgetting)
MAX_ACCURACY_DROP = 0.02    # reject corrections if test accuracy falls more than this

SPLITS_PATH = Path("splits.json")
CLASS_NAMES_PATH = Path("class_names.json")
MODEL_PATH = Path("trash_classifier_resnet50.keras")
QUEUE_PATH = Path("training_queue.json")
ARCHIVE_PATH = Path("corrections_archive.json")  # corrections already learned; reused in full retrains

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# HELPERS
# ============================================================

def banner(title):
    print(f"\n{'=' * 40}\n{title}\n{'=' * 40}")


def load_json(path, default=None):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    if default is not None:
        return default
    raise FileNotFoundError(f"Required file not found: {path}")


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def clean_items(items, num_classes, name):
    """Drop entries with missing files or out-of-range labels."""
    good = [
        item for item in items
        if Path(item["path"]).is_file() and 0 <= int(item["label"]) < num_classes
    ]
    dropped = len(items) - len(good)
    if dropped:
        print(f"WARNING: dropped {dropped} invalid entries from {name} (missing file or bad label)")
    return good


# ============================================================
# DATA
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, tf.cast(label, tf.int32)


def make_dataset(items, shuffle=False, cache=False):
    paths = [item["path"] for item in items]
    labels = [int(item["label"]) for item in items]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Shuffle cheap (path, label) pairs BEFORE decoding images,
    # so the buffer can cover the whole dataset without eating RAM.
    if shuffle:
        dataset = dataset.shuffle(len(items), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def compute_class_weights(items, num_classes):
    counts = np.bincount([int(i["label"]) for i in items], minlength=num_classes)
    total = counts.sum()
    return {c: total / (num_classes * max(int(n), 1)) for c, n in enumerate(counts)}


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="augmentation")

    base_model = tf.keras.applications.ResNet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)  # keeps BatchNorm in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs), base_model


def find_base_model(model):
    """Locate the nested ResNet50 inside a loaded model."""
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model)
                and layer.name != "augmentation")


def unfreeze_tail(base_model, n_layers):
    """Unfreeze the last n layers, but keep BatchNorm layers frozen."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
        ),
    ]


# ============================================================
# EVALUATION
# ============================================================

def report_weakest_classes(model, dataset, class_names, worst=5):
    y_true = np.concatenate([y.numpy() for _, y in dataset])
    y_pred = np.argmax(model.predict(dataset, verbose=0), axis=1)

    n = len(class_names)
    totals = np.bincount(y_true, minlength=n)
    correct = np.bincount(y_true[y_pred == y_true], minlength=n)
    accuracy = correct / np.maximum(totals, 1)

    ranked = [c for c in np.argsort(accuracy) if totals[c] > 0][:worst]
    print(f"\nWeakest {len(ranked)} classes on test set:")
    for c in ranked:
        print(f"  {class_names[c]:<30} {accuracy[c]:.1%}  ({correct[c]}/{totals[c]})")


# ============================================================
# TRAINING PHASES
# ============================================================

def train_from_scratch(num_classes, train_items, train_ds, val_ds):
    model, base_model = build_model(num_classes)
    class_weight = compute_class_weights(train_items, num_classes) if USE_CLASS_WEIGHTS else None

    banner("PHASE 1: Train classifier head")
    compile_model(model, LR_INITIAL)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_INITIAL,
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )

    banner("PHASE 2: Fine-tune ResNet50")
    unfreeze_tail(base_model, UNFREEZE_LAYERS)
    compile_model(model, LR_FINE_TUNE)  # recompile after changing trainable flags
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )

    return model, base_model


def train_corrections(model, base_model, queue, train_items, class_names):
    banner("PHASE 3: Correction queue")
    print(f"Corrections waiting: {len(queue)}")
    for item in queue:
        print(f"  {item['path']} -> {class_names[int(item['label'])]}")

    # Repeat BEFORE batching so every repeat lands in a different, shuffled batch.
    corrections = queue * CORRECTION_REPEAT

    # Mix in a random sample of original training data so the model
    # doesn't overfit the few corrections and forget everything else.
    rng = np.random.default_rng(SEED)
    replay_n = min(len(train_items), len(queue) * REPLAY_PER_CORRECTION)
    replay_idx = rng.choice(len(train_items), size=replay_n, replace=False)
    replay = [train_items[i] for i in replay_idx]

    mixed = corrections + replay
    print(f"Training on {len(corrections)} correction samples + {len(replay)} replay samples")

    correction_ds = make_dataset(mixed, shuffle=True)

    unfreeze_tail(base_model, UNFREEZE_LAYERS)
    compile_model(model, LR_CORRECTION)
    model.fit(correction_ds, epochs=EPOCHS_CORRECTION)


# ============================================================
# MAIN
# ============================================================

def choose_mode(queue_count, model_exists):
    """Ask what to do. Returns True for new-data-only, False for full retrain, None to quit."""
    if not model_exists:
        print("No saved model yet, so running a full training.")
        return False

    print("\nWhat do you want to do?")
    print(f"  1) Train on new data only ({queue_count} corrected image(s) waiting). Fast, keeps the current model.")
    print("  2) Full retrain from scratch. Slow.")
    print("  q) Quit")
    while True:
        choice = input("Enter 1, 2 or q: ").strip().lower()
        if choice == "1":
            return True
        if choice == "2":
            return False
        if choice == "q":
            return None


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--corrections-only",
        action="store_true",
        help="Skip phases 1-2; load the saved model and only train on the correction queue.",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Full retrain from scratch (skips the menu).",
    )
    args = parser.parse_args()

    # ---------- pick mode ----------
    queue_count = len(load_json(QUEUE_PATH, default=[]))
    if args.corrections_only:
        corrections_only = True
    elif args.full:
        corrections_only = False
    else:
        corrections_only = choose_mode(queue_count, MODEL_PATH.exists())
        if corrections_only is None:
            return

    if corrections_only and queue_count == 0:
        print("No new data in training_queue.json. Use test.py with --correct to add some.")
        return

    tf.keras.utils.set_random_seed(SEED)

    # ---------- load data ----------
    splits = load_json(SPLITS_PATH)
    class_names = load_json(CLASS_NAMES_PATH)
    num_classes = len(class_names)

    archive = clean_items(load_json(ARCHIVE_PATH, default=[]), num_classes, "archive")
    train_items = clean_items(splits["train"], num_classes, "train") + archive
    val_items = clean_items(splits["validation"], num_classes, "validation")
    test_items = clean_items(splits["test"], num_classes, "test")

    print(f"Classes: {num_classes} | train: {len(train_items)} "
          f"(incl. {len(archive)} past corrections) | val: {len(val_items)} | test: {len(test_items)}")

    val_ds = make_dataset(val_items, cache=True)
    test_ds = make_dataset(test_items, cache=True)

    # ---------- phases 1 & 2 ----------
    if corrections_only:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found. Run a full training first.")
        model = tf.keras.models.load_model(MODEL_PATH)
        base_model = find_base_model(model)
    else:
        train_ds = make_dataset(train_items, shuffle=True)
        model, base_model = train_from_scratch(num_classes, train_items, train_ds, val_ds)
        model.save(MODEL_PATH)  # save before corrections so we can safely reject them

    # ---------- baseline ----------
    banner("TEST (before corrections)")
    baseline_loss, baseline_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {baseline_acc:.4f}")
    report_weakest_classes(model, test_ds, class_names)

    # ---------- phase 3 ----------
    queue = clean_items(load_json(QUEUE_PATH, default=[]), num_classes, "queue")

    if not queue:
        banner("PHASE 3: No correction images")
        print(f"\nModel saved as {MODEL_PATH}")
        print("\nTraining complete.")
        return

    train_corrections(model, base_model, queue, train_items, class_names)

    banner("TEST (after corrections)")
    _, new_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {new_acc:.4f} (was {baseline_acc:.4f})")

    if baseline_acc - new_acc > MAX_ACCURACY_DROP:
        print(f"\nAccuracy dropped more than {MAX_ACCURACY_DROP:.0%}. "
              f"Corrections NOT saved; queue left untouched.")
        return

    # ---------- save & clear queue ----------
    model.save(MODEL_PATH)
    print(f"\nModel saved as {MODEL_PATH}")

    save_json(ARCHIVE_PATH, archive + queue)  # remember them for future full retrains
    save_json(QUEUE_PATH, [])
    print("Corrections archived and queue cleared.")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()