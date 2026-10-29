import json
import tensorflow as tf

# =========================
# Settings
# =========================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
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
# Create TensorFlow dataset
# =========================

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


# =========================
# MobileNetV2
# =========================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Don't change MobileNetV2's existing weights
base_model.trainable = False


# =========================
# Our classifier
# =========================

model = tf.keras.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# =========================
# Configure model
# =========================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# =========================
# Train
# =========================

model.summary()

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS
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