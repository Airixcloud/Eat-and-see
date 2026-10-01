import json
import sys
import predict
import random
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
with open("splits.json", "r") as f:
    images = json.load(f)

test_images = images["test"]

random_sample = random.sample(test_images, 10)

for image in random_sample:
    print(f"Predicting for image: {image['path']}")
    predict.predict(image["path"], model, class_names, category_map)