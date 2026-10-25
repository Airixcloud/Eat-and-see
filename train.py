import json
import tensorflow as tf

# =========================
# Settings
# =========================

IMAGE_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
NUM_CLASSES = 30

# =========================
# Load data
# =========================

with open("splits.json", "r") as f:
    splits = json.load(f)

with open("class_names.json", "r") as f:
    class_names = json.load(f)


# =========================
# Load and preprocess image
# =========================

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
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )
    return image, label


# =========================
# Data Augmentation (Tăng cường dữ liệu)
# Giúp mô hình nhận diện tốt khi vật thể bị xoay góc, nghiêng, lệch tâm hay đổi màu
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.15),
], name="data_augmentation")


# =========================
# Create TensorFlow dataset
# =========================

def make_dataset(data, shuffle=False):
    paths = [item["path"] for item in data]
    labels = [item["label"] for item in data]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # splits.json is sorted by class, so shuffle the whole list (before decoding, cheap)
    # otherwise each batch only contains 2-3 classes and the model forgets earlier classes
    if shuffle:
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

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


# =========================
# MobileNetV2 Base Model
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Giai đoạn 1: Đóng băng base model để huấn luyện classifier head
base_model.trainable = False


# =========================
# Build Model Architecture
# =========================

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)


# =========================
# Giai đoạn 1: Feature Extraction (Warm-up)
# =========================

print("\n--- GIAI ĐOẠN 1: HUẤN LUYỆN CLASSIFIER HEAD ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks_phase1 = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

history_phase1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_phase1
)


# =========================
# Giai đoạn 2: Fine-Tuning các layer trên cùng của MobileNetV2
# Giúp mô hình học sâu hơn về chất liệu bề mặt thật (giấy vs nilon vs xốp)
# =========================

print("\n--- GIAI ĐOẠN 2: FINE-TUNING MOBILENETV2 ---")
base_model.trainable = True

# Đóng băng 100 layer đầu (trích xuất cạnh cơ bản), chỉ mở khóa các layer đặc trưng bậc cao
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Sử dụng Learning Rate nhỏ để không phá vỡ trọng số đã hội tụ
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
    tf.keras.callbacks.ModelCheckpoint("trash_classifier.keras", monitor="val_accuracy", save_best_only=True)
]

history_phase2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history_phase1.epoch[-1] + 1,
    callbacks=callbacks_phase2
)


# =========================
# Test
# =========================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print()
print("Test accuracy:", test_accuracy)


# =========================
# Save model
# =========================

model.save("trash_classifier.keras")

print()
print("Model saved as trash_classifier.keras")