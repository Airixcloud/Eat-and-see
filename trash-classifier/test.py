"""
Run the trash classifier on images and (optionally) queue corrections for retraining.

Usage:
    python test.py                                  # interactive menu (corrections are on for single/folder)
    python test.py single photo.jpg --correct       # one image, queue a fix if wrong
    python test.py batch --folder new_images --correct
                                                               # a folder of NEW images, queue fixes
    python test.py batch --count 20                 # random test-split sample (labels known)

After queueing corrections, train on them with:
    python train.py   (then choose option 1: train on new data only)
"""

import argparse
import json
import os
import random
from pathlib import Path

import tensorflow as tf

import predict

BASE_DIR = Path(__file__).resolve().parent  # project folder (model, splits, etc. live here)
LAUNCH_DIR = Path.cwd()                      # where the user ran the command from

MODEL_PATH = Path("trash_classifier_resnet50.keras")
CLASS_NAMES_PATH = Path("class_names.json")
SPLITS_PATH = Path("splits.json")
DEFAULT_TEST_SAMPLE = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Waste category mapping
CATEGORY_MAP = {
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


def user_path(text):
    """Resolve a path the user typed, relative to where they launched the script.
    Also handles drag-and-drop paths (quotes / escaped spaces) from the Mac terminal."""
    text = text.strip().strip("'\"").replace("\\ ", " ")
    return str(LAUNCH_DIR / Path(text).expanduser())


def load_resources():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Train it first.")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    # Fail early if a class has no category instead of crashing mid-prediction
    missing = [name for name in class_names if name not in CATEGORY_MAP]
    if missing:
        raise SystemExit(f"Classes missing from CATEGORY_MAP: {', '.join(missing)}")

    return model, class_names


def load_splits():
    with open(SPLITS_PATH, "r") as f:
        return json.load(f)


def held_out_paths():
    """Validation + test images. These must never be queued for training."""
    splits = load_splits()
    items = splits.get("validation", []) + splits.get("test", [])
    return {str(Path(item["path"]).resolve()) for item in items}


# ============================================================
# MODES
# ============================================================

def predict_single(image_path, model, class_names, ask_correction=False):
    if not Path(image_path).is_file():
        print(f"File not found: {image_path}")
        return

    if ask_correction and str(Path(image_path).resolve()) in held_out_paths():
        print("(This image is in your validation/test split, so it won't be queued.)")
        ask_correction = False

    predict.predict(image_path, model, class_names, CATEGORY_MAP, ask_correction=ask_correction)


def predict_folder(folder, model, class_names, ask_correction=False):
    """Run on a folder of NEW images; optionally queue corrections."""
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {folder}")
        return

    held_out = held_out_paths() if ask_correction else set()

    for i, path in enumerate(images, start=1):
        print(f"\n[{i}/{len(images)}] {path}")
        allow = ask_correction and str(path.resolve()) not in held_out
        if ask_correction and not allow:
            print("  (in validation/test split, not queued)")
        try:
            predict.predict(str(path), model, class_names, CATEGORY_MAP, ask_correction=allow)
        except Exception as error:  # one bad image shouldn't stop the run
            print(f"  Skipped ({type(error).__name__}: {error})")


def predict_test_sample(count, model, class_names):
    """Random test-split images. Labels are known, so show right/wrong (no queueing)."""
    test_items = load_splits()["test"]
    sample = random.sample(test_items, min(count, len(test_items)))

    evaluated = correct = 0
    for i, item in enumerate(sample, start=1):
        print(f"\n[{i}/{len(sample)}] {item['path']}")
        try:
            results = predict.predict(item["path"], model, class_names, CATEGORY_MAP)
        except Exception as error:
            print(f"  Skipped ({type(error).__name__}: {error})")
            continue

        true_name = class_names[int(item["label"])]
        hit = results[0]["class"] == true_name
        evaluated += 1
        correct += hit
        print(f"True label: {true_name} -> {'CORRECT' if hit else 'WRONG'}")

    if evaluated:
        print(f"\nSample accuracy: {correct}/{evaluated} ({correct / evaluated:.0%})")


def prompt_action():
    print("\nWhat do you want to test?")
    print("  1) One image (you can correct the prediction)")
    print("  2) A folder of new images (you can correct the predictions)")
    print("  3) Random sample from the test split (checks accuracy, no corrections)")
    print("  q) Quit")
    actions = {"1": "single", "2": "folder", "3": "sample", "q": "quit"}
    while True:
        choice = input("Enter 1, 2, 3 or q: ").strip().lower()
        if choice in actions:
            return actions[choice]
        print("Please enter 1, 2, 3 or q.")


def main():
    os.chdir(BASE_DIR)  # so model/splits/queue files are found no matter where you run from

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="mode")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--correct",
        action="store_true",
        help="After each prediction, offer to queue a correction for training_queue.json",
    )

    single = subparsers.add_parser("single", parents=[common], help="Predict one image")
    single.add_argument("image_path", nargs="?")

    batch = subparsers.add_parser("batch", parents=[common], help="Predict a folder or a test sample")
    batch.add_argument("--folder", help="Folder of new images to predict (use with --correct)")
    batch.add_argument("--count", type=int, default=DEFAULT_TEST_SAMPLE,
                       help="Size of the random test-split sample (ignored with --folder)")

    args = parser.parse_args()

    # ---------- decide what to do ----------
    if args.mode is None:
        # Interactive: corrections are ON for single images and folders
        action = prompt_action()
        if action == "quit":
            return
        ask_correction = action in ("single", "folder")
        target = None
        count = DEFAULT_TEST_SAMPLE
    elif args.mode == "single":
        action, ask_correction, target, count = "single", args.correct, args.image_path, DEFAULT_TEST_SAMPLE
    else:
        action = "folder" if args.folder else "sample"
        ask_correction, target, count = args.correct, args.folder, args.count

    model, class_names = load_resources()

    if action == "single":
        image_path = user_path(target or input("Enter the path to the image: "))
        predict_single(image_path, model, class_names, ask_correction)
    elif action == "folder":
        folder = user_path(target or input("Enter the path to the folder: "))
        predict_folder(folder, model, class_names, ask_correction)
    else:
        if ask_correction:
            print("Note: test-split images already have labels, so --correct only applies with --folder.")
        predict_test_sample(count, model, class_names)


if __name__ == "__main__":
    main()