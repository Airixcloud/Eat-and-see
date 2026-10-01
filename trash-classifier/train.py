import json
import tensorflow as tf

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Phase 1: train classifier
EPOCHS_INITIAL = 5

# Phase 2: fine-tune MobileNetV2
EPOCHS_FINE_TUNE = 5

NUM_CLASSES = 30


# -------------------------
# Load split information
# -------------------------

with open("splits.json", "r") as f:
    splits = json.load(f)

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# -------------------------
# Load images
# -------------------------

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


def make_dataset(data, shuffle=False):
    paths = [item["path"] for item in data]
    labels = [item["label"] for item in data]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = make_dataset(
    splits["train"],
    shuffle=True
)

validation_dataset = make_dataset(
    splits["validation"]
)

test_dataset = make_dataset(
    splits["test"]
)


# -------------------------
# Data augmentation
# -------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# -------------------------
# MobileNetV2
# -------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Phase 1: freeze MobileNetV2
base_model.trainable = False


# -------------------------
# Build model
# -------------------------

inputs = tf.keras.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# -------------------------
# Phase 1
# -------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n===== PHASE 1 =====")
print("Training classifier...")

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_INITIAL
)


# -------------------------
# Phase 2: Fine-tuning
# -------------------------

print("\n===== PHASE 2 =====")
print("Fine-tuning MobileNetV2...")

base_model.trainable = True


# Freeze most of MobileNetV2.
# Only train the last 30 layers.

for layer in base_model.layers[:-30]:
    layer.trainable = False


# Use a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_FINE_TUNE
)


# -------------------------
# Test
# -------------------------

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print()
print("Test accuracy:", test_accuracy)


# -------------------------
# Save model
# -------------------------

model.save("trash_classifier.keras")

print()
print("Model saved as trash_classifier.keras")