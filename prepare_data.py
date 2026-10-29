from pathlib import Path
import random
import json

DATASET_DIR = Path("images")
SEED = 42

random.seed(SEED)

# Find the 30 class folders
classes = sorted(
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}

# -------------------------
# Create class_names.json
# -------------------------

with open("class_names.json", "w") as f:
    json.dump(classes, f, indent=4)


# -------------------------
# Create train/validation/test splits
# -------------------------

train = []
validation = []
test = []

for class_name in classes:

    class_index = class_to_index[class_name]

    default_images = list(
        (DATASET_DIR / class_name / "default").glob("*")
    )

    real_world_images = list(
        (DATASET_DIR / class_name / "real_world").glob("*")
    )

    random.shuffle(default_images)
    random.shuffle(real_world_images)

    # Default images
    train_default = default_images[:200]
    validation_default = default_images[200:]

    # Real-world images
    train_real_world = real_world_images[:200]
    test_real_world = real_world_images[200:]

    # Training
    for path in train_default + train_real_world:
        train.append({
            "path": str(path),
            "label": class_index
        })

    # Validation
    for path in validation_default:
        validation.append({
            "path": str(path),
            "label": class_index
        })

    # Testing
    for path in test_real_world:
        test.append({
            "path": str(path),
            "label": class_index
        })


# -------------------------
# Save splits.json
# -------------------------

splits = {
    "train": train,
    "validation": validation,
    "test": test
}

with open("splits.json", "w") as f:
    json.dump(splits, f, indent=4)


# -------------------------
# Print results
# -------------------------

print("Classes:", len(classes))
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

print("\nClass mapping:")

for name, index in class_to_index.items():
    print(index, "->", name)