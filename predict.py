import json
import sys
import tensorflow as tf

IMAGE_SIZE = 224

# Load model
model = tf.keras.models.load_model("trash_classifier.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Waste category mapping
category_map = {
    "aerosol_cans": "recyclable",
    "aluminum_food_cans": "recyclable",
    "aluminum_soda_cans": "recyclable",
    "cardboard_boxes": "recyclable",
    "cardboard_packaging": "recyclable",
    "clothing": "recyclable",
    "coffee_grounds": "organic",
    "disposable_plastic_cutlery": "non-recyclable",
    "eggshells": "organic",
    "food_waste": "organic",
    "glass_beverage_bottles": "recyclable",
    "glass_cosmetic_containers": "recyclable",
    "glass_food_jars": "recyclable",
    "magazines": "recyclable",
    "newspaper": "recyclable",
    "office_paper": "recyclable",
    "paper_cups": "non-recyclable",
    "plastic_cup_lids": "recyclable",
    "plastic_detergent_bottles": "recyclable",
    "plastic_food_containers": "recyclable",
    "plastic_shopping_bags": "recyclable",
    "plastic_soda_bottles": "recyclable",
    "plastic_straws": "non-recyclable",
    "plastic_trash_bags": "non-recyclable",
    "plastic_water_bottles": "recyclable",
    "shoes": "recyclable",
    "steel_food_cans": "recyclable",
    "styrofoam_cups": "non-recyclable",
    "styrofoam_food_containers": "non-recyclable",
    "tea_bags": "organic",
}


def predict(image_path):
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

    predicted_index = tf.argmax(predictions[0]).numpy()
    confidence = predictions[0][predicted_index]

    item = class_names[predicted_index]
    category = category_map.get(item, "unknown")

    print()
    print("Prediction:", item)
    print("Confidence:", f"{confidence * 100:.2f}%")
    print("Waste category:", category)


if len(sys.argv) != 2:
    print("Usage:")
    print("uv run python predict.py <image_path>")
    sys.exit(1)

predict(sys.argv[1])