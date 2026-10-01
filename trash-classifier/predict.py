import tensorflow as tf

IMAGE_SIZE = 224

def predict(image_path, model, class_names, category_map):
    image = tf.io.read_file(image_path)

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

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    # Get the top 10 predictions
    top_10 = tf.argsort(
        predictions[0],
        direction="DESCENDING"
    )[:10]

    print()
    print("Top 10 predictions:")

    for rank, index in enumerate(top_10, start=1):
        index = index.numpy()
        confidence = predictions[0][index]

        item = class_names[index]
        category = category_map.get(item, "unknown")

        print(
            f"{rank}. {item} - "
            f"{confidence * 100:.2f}% "
            f"({category})"
        )
