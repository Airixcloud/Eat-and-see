import json, random, time
from collections import Counter
import cv2
import numpy as np
import tensorflow as tf

PER_CLASS = 10  # moi loai lay 10 anh real_world (test) + 10 anh default (validation) = 600 anh
SEED = 42

model = tf.keras.models.load_model("trash_classifier.keras")
names = json.load(open("class_names.json"))
splits = json.load(open("splits.json"))

CAT = {
    "aerosol_cans": "recyclable", "aluminum_food_cans": "recyclable", "aluminum_soda_cans": "recyclable",
    "cardboard_boxes": "recyclable", "cardboard_packaging": "recyclable", "clothing": "recyclable",
    "coffee_grounds": "organic", "disposable_plastic_cutlery": "non-recyclable", "eggshells": "organic",
    "food_waste": "organic", "glass_beverage_bottles": "recyclable", "glass_cosmetic_containers": "recyclable",
    "glass_food_jars": "recyclable", "magazines": "recyclable", "newspaper": "recyclable",
    "office_paper": "recyclable", "paper_cups": "non-recyclable", "plastic_cup_lids": "recyclable",
    "plastic_detergent_bottles": "recyclable", "plastic_food_containers": "recyclable",
    "plastic_shopping_bags": "recyclable", "plastic_soda_bottles": "recyclable", "plastic_straws": "non-recyclable",
    "plastic_trash_bags": "non-recyclable", "plastic_water_bottles": "recyclable", "shoes": "recyclable",
    "steel_food_cans": "recyclable", "styrofoam_cups": "non-recyclable", "styrofoam_food_containers": "non-recyclable",
    "tea_bags": "organic",
}


def sample(items):
    rng = random.Random(SEED)
    by_cls = {}
    for it in items:
        by_cls.setdefault(it["label"], []).append(it)
    out = []
    for lbl in sorted(by_cls):
        out += rng.sample(by_cls[lbl], min(PER_CLASS, len(by_cls[lbl])))
    return out


def predict_stretch(rgb_list):
    batch = np.stack([cv2.resize(im, (224, 224)) for im in rgb_list]).astype(np.float32)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return np.concatenate([model.predict(batch[i:i + 64], verbose=0) for i in range(0, len(batch), 64)])


def evaluate(items, label):
    t = time.time()
    imgs, ys = [], []
    for it in items:
        bgr = cv2.imread(it["path"])
        if bgr is not None:
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ys.append(it["label"])
    ys = np.array(ys)
    probs = predict_stretch(imgs)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == ys
    cat_ok = np.array([CAT[names[p]] == CAT[names[y]] for p, y in zip(pred, ys)])

    print(f"\n=== {label}: {len(ys)} anh ({time.time() - t:.0f}s) ===")
    print(f"Dung ten vat the : {correct.mean() * 100:.1f}%")
    print(f"Dung nhom rac    : {cat_ok.mean() * 100:.1f}%")
    print(f"Do tu tin TB khi DUNG: {conf[correct].mean() * 100:.1f}% | khi SAI: {conf[~correct].mean() * 100:.1f}%")
    wrong_hi = (~correct & (conf >= 0.8)).sum()
    print(f"So anh SAI ma do tu tin >= 80%: {wrong_hi}/{(~correct).sum()}")

    per_cls = {names[c]: correct[ys == c].mean() for c in np.unique(ys)}
    worst = sorted(per_cls.items(), key=lambda kv: kv[1])[:8]
    print("Loai kem nhat: " + ", ".join(f"{n} {a * 100:.0f}%" for n, a in worst))

    pairs = Counter((names[y], names[p]) for y, p in zip(ys, pred) if y != p)
    print("Cap hay nham nhat (that -> doan):")
    for (a, b), n in pairs.most_common(10):
        print(f"   {n:2d} x  {a} -> {b}")
    return correct, cat_ok


c1, k1 = evaluate(sample(splits["test"]), "TEST real_world (giong anh thuc te/camera)")
c2, k2 = evaluate(sample(splits["validation"]), "VALIDATION default (anh studio)")
print(f"\n=== TONG 600 anh: ten vat the {np.concatenate([c1, c2]).mean() * 100:.1f}% | "
      f"nhom rac {np.concatenate([k1, k2]).mean() * 100:.1f}% ===")
