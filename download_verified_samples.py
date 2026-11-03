import os
import io
import json
import urllib.request
import urllib.parse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "web", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Specific, targeted queries for real waste/recycling objects
TARGETS = {
    "glass": [
        "Glass bottle recycling conveyer",
        "Soju bottles in a glass recycling bin",
        "Empty beer bottles crate"
    ],
    "plastic_bottles": [
        "An empty bottle water plastic",
        "Empty plastic bottle",
        "Crushed plastic water bottles"
    ],
    "metal_cans": [
        "Crushed beverage cans",
        "Aluminum beverage can",
        "Tin cans recycling"
    ],
    "cardboard": [
        "Cardboard boxes recycling baler",
        "Corrugated cardboard box",
        "Stacked cardboard boxes"
    ],
    "paper": [
        "Waste paper bundle",
        "Newspaper stack recycling",
        "Discarded newspapers"
    ],
    "clothing_shoes": [
        "Clothing donation bin",
        "Worn out sneakers shoes",
        "Used clothes textile recycling"
    ],
    "food_waste": [
        "Food waste compost bin",
        "Vegetable scraps compost",
        "Kitchen food scraps"
    ],
    "coffee_tea": [
        "Used coffee grounds",
        "Used tea bag",
        "Espresso coffee puck"
    ],
    "eggshells": [
        "Broken eggshells compost",
        "Cracked eggshells",
        "Egg shells kitchen waste"
    ],
    "styrofoam": [
        "Polystyrene foam food container",
        "Styrofoam takeout container",
        "Expanded polystyrene foam"
    ],
    "single_use_plastic": [
        "Disposable plastic cutlery",
        "Plastic drinking straw litter",
        "Single-use plastic cups"
    ],
    "plastic_bags": [
        "Plastic shopping bag litter",
        "Plastic carrier bag waste",
        "Plastic bag recycling"
    ],
    "phone_tablet": [
        "Electronic waste mobile phones",
        "Old cell phones e-waste",
        "Old broken smartphone"
    ],
    "laptop_computer": [
        "Old broken laptop computer",
        "Laptop computer e-waste",
        "Discarded laptop"
    ],
    "mouse_keyboard": [
        "Computer keyboard hardware",
        "Computer mouse old",
        "Broken computer keyboard"
    ],
    "tv_appliances": [
        "Electronic waste CRT television",
        "Old microwave oven",
        "Discarded television monitor"
    ]
}

HEADERS = {"User-Agent": "EcoSortBot/1.0 (contact@ecosort.ai)"}

def search_wikimedia(query):
    api_url = "https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url|mime|size&generator=search&gsrnamespace=6&gsrlimit=5&gsrsearch=" + urllib.parse.quote(query)
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        data = json.loads(urllib.request.urlopen(req, timeout=8).read().decode('utf-8'))
        pages = data.get("query", {}).get("pages", {})
        results = []
        for pid, pdata in pages.items():
            info = pdata.get("imageinfo", [{}])[0]
            url = info.get("url", "")
            mime = info.get("mime", "")
            size = info.get("size", 0)
            if url and size > 20000 and ("jpeg" in mime or "png" in mime or "jpg" in mime):
                results.append(url)
        return results
    except Exception as e:
        return []

def download_and_save(url, dest_path):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=10).read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        
        # Center-crop to 4:3 ratio and resize to 600x450
        target_w, target_h = 600, 450
        w, h = img.size
        target_ratio = target_w / target_h
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
            
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img.save(dest_path, "JPEG", quality=88)
        return True
    except Exception as e:
        return False

print("Starting verified image collection from Wikimedia Commons...")

for cat, q_list in TARGETS.items():
    success_count = 0
    for idx, query in enumerate(q_list, start=1):
        if success_count >= 2:
            break
        dest_filename = f"{cat}_{success_count + 1}.jpg"
        dest_path = os.path.join(SAMPLES_DIR, dest_filename)
        
        urls = search_wikimedia(query)
        downloaded = False
        for url in urls:
            if download_and_save(url, dest_path):
                downloaded = True
                success_count += 1
                print(f"[OK] {dest_filename} downloaded for '{cat}'")
                break

print("Image collection finished.")
