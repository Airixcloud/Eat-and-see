import os, glob, shutil
from PIL import Image

BRAIN_DIR = r"C:\Users\GIAKHOI\.gemini\antigravity\brain\febae2d8-f435-4b7a-8130-544bcc2a37b5"
SAMPLES_DIR = r"C:\Users\GIAKHOI\Desktop\VGU\trash-classifier\trash-classifier\web\samples"

targets = [
    "metal_cans_1", "metal_cans_2",
    "cardboard_1", "cardboard_2",
    "paper_1", "paper_2",
    "glass_1", "glass_2",
    "clothing_shoes_1", "clothing_shoes_2",
    "food_waste_1", "food_waste_2",
    "coffee_tea_1", "coffee_tea_2",
    "eggshells_1", "eggshells_2",
    "styrofoam_1", "styrofoam_2",
    "single_use_plastic_1", "single_use_plastic_2",
    "plastic_bags_1", "plastic_bags_2",
    "plastic_bottles_1", "plastic_bottles_2",
    "phone_tablet_1", "phone_tablet_2",
    "laptop_1", "laptop_2",
    "mouse_keyboard_1", "mouse_keyboard_2",
    "tv_appliances_1", "tv_appliances_2"
]

print("Scanning for generated images in:", BRAIN_DIR)
for t in targets:
    pattern = os.path.join(BRAIN_DIR, f"{t}*.jpg")
    matches = glob.glob(pattern)
    if matches:
        latest = max(matches, key=os.path.getmtime)
        dest = os.path.join(SAMPLES_DIR, f"{t}.jpg")
        img = Image.open(latest).convert("RGB")
        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        img.save(dest, "JPEG", quality=90)
        print(f"Copied & optimized: {os.path.basename(latest)} -> {t}.jpg ({img.size})")
