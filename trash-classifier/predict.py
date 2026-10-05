"""
Prediction helpers for the trash classifier.

Preprocessing here MUST match train.py (ResNet50 preprocessing).
Misclassified images can be added to training_queue.json, which the training
script trains on when you choose "new data only".
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_SIZE = 224
TOP_K = 20
QUEUE_PATH = Path("training_queue.json")


# ============================================================
# PREPROCESSING (same as training)
# ============================================================

def load_image(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return tf.expand_dims(image, axis=0)  # add batch dimension


# ============================================================
# CORRECTION QUEUE
# ============================================================

def add_to_queue(image_path, label, queue_path=QUEUE_PATH):
    """Add (or update) a corrected image in the training queue."""
    path = str(Path(image_path).resolve())

    queue = []
    if queue_path.exists():
        with open(queue_path, "r") as f:
            queue = json.load(f)

    # One entry per image: a newer correction replaces an older one
    queue = [item for item in queue if item["path"] != path]
    queue.append({"path": path, "label": int(label)})

    with open(queue_path, "w") as f:
        json.dump(queue, f, indent=4)

    return len(queue)


def prompt_for_correction(image_path, class_names, results):
    """Ask whether the top prediction is right; queue a correction if not."""
    answer = input(
        "Correct? [Enter = yes | rank 1-{} | class name | s = skip]: ".format(len(results))
    ).strip()

    if answer == "" or answer == "1" or answer.lower() == "s":
        return

    if answer.isdigit() and 1 <= int(answer) <= len(results):
        label = results[int(answer) - 1]["index"]
    elif answer in class_names:
        label = class_names.index(answer)
    else:
        print("Not recognised, skipping correction.")
        return

    size = add_to_queue(image_path, label)
    print(f"Queued as '{class_names[label]}' ({size} correction(s) waiting).")


# ============================================================
# PREDICT
# ============================================================

def predict(image_path, model, class_names, category_map, top_k=TOP_K, ask_correction=False):
    """Print the top predictions and return them as a list of dicts."""
    probabilities = model.predict(load_image(image_path), verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for index in top_indices:
        name = class_names[index]
        results.append({
            "index": int(index),
            "class": name,
            "confidence": float(probabilities[index]),
            "category": category_map.get(name, "unknown"),
        })

    print()
    print(f"Top {len(results)} predictions:")
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. {r['class']} - {r['confidence'] * 100:.2f}% ({r['category']})")

    if ask_correction:
        prompt_for_correction(image_path, class_names, results)

    return results