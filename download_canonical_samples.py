import os
import io
import urllib.request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "web", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Curated, unambiguous direct image URLs for every single waste category
CANONICAL_SAMPLES = {
    # 1. RÁC TÁI CHẾ
    "glass_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/1/11/Beer_bottles_2018_G1.jpg",
    "glass_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/48/Cocacolacollection.JPG",
    
    "plastic_bottles_1.jpg": "LOCAL_test_bottle.jpg",
    "plastic_bottles_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Antifreeze.jpg",
    
    "metal_cans_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Drinking_can_ring-pull_tab.jpg",
    "metal_cans_2.jpg": "https://images.unsplash.com/photo-1535982330050-f1c2fb79ff78?w=600&auto=format&fit=crop&q=80",
    
    "cardboard_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/eb/Box.agr.jpg",
    "cardboard_2.jpg": "https://images.unsplash.com/photo-1530587191325-3db32d826c18?w=600&auto=format&fit=crop&q=80",
    
    "paper_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/75/NYTimes-Page1-11-11-1918.jpg",
    "paper_2.jpg": "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=600&auto=format&fit=crop&q=80",
    
    "clothing_shoes_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/9/96/Clothes.jpg",
    "clothing_shoes_2.jpg": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&auto=format&fit=crop&q=80",

    # 2. RÁC HỮU CƠ
    "food_waste_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/e5/Compost_site_germany.JPG",
    "food_waste_2.jpg": "https://images.unsplash.com/photo-1610484826967-09c5720778c7?w=600&auto=format&fit=crop&q=80",
    
    "coffee_tea_1.jpg": "https://images.unsplash.com/photo-1511920170033-f8396924c348?w=600&auto=format&fit=crop&q=80",
    "coffee_tea_2.jpg": "https://images.unsplash.com/photo-1576092768241-dec231879fc3?w=600&auto=format&fit=crop&q=80",
    
    "eggshells_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b1/Egg-rmh.jpg",
    "eggshells_2.jpg": "https://images.unsplash.com/photo-1569288052389-dac9b01c9c05?w=600&auto=format&fit=crop&q=80",

    # 3. RÁC VÔ CƠ / KHÔNG TÁI CHẾ
    "styrofoam_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Egg_breakfast_sandwich.jpg",
    "styrofoam_2.jpg": "https://images.unsplash.com/photo-1605600659908-0ef719419d41?w=600&auto=format&fit=crop&q=80",
    
    "single_use_plastic_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/0/02/Paper_cup.jpeg",
    "single_use_plastic_2.jpg": "https://images.unsplash.com/photo-1582407947304-fd86f028f716?w=600&auto=format&fit=crop&q=80",
    
    "plastic_bags_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Trash_bin_in_Paris.jpg",
    "plastic_bags_2.jpg": "https://images.unsplash.com/photo-1597484661643-2f5fef640dd1?w=600&auto=format&fit=crop&q=80",

    # 4. RÁC ĐIỆN TỬ
    "phone_tablet_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/30/Ewaste-pile.jpg",
    "phone_tablet_2.jpg": "https://images.unsplash.com/photo-1584438784894-089d6a62b8fa?w=600&auto=format&fit=crop&q=80",
    
    "laptop_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/6/68/Laptop_collage.jpg",
    "laptop_2.jpg": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=600&auto=format&fit=crop&q=80",
    
    "mouse_keyboard_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/2/22/3-Tasten-Maus_Microsoft.jpg",
    "mouse_keyboard_2.jpg": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=600&auto=format&fit=crop&q=80",
    
    "tv_appliances_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/40/Cptvdisplay.jpg",
    "tv_appliances_2.jpg": "https://images.unsplash.com/photo-1593784991095-a205069470b6?w=600&auto=format&fit=crop&q=80",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

def process_and_save(img, dest_path):
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

for fname, url in CANONICAL_SAMPLES.items():
    dest_path = os.path.join(SAMPLES_DIR, fname)
    if url == "LOCAL_test_bottle.jpg":
        src_path = os.path.join(BASE_DIR, "test_bottle.jpg")
        img = Image.open(src_path).convert("RGB")
        process_and_save(img, dest_path)
        print(f"[OK Local] {fname}")
        continue
        
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=12).read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        process_and_save(img, dest_path)
        print(f"[OK Download] {fname}")
    except Exception as e:
        print(f"[ERR] {fname}: {e}")

print("Canonical sample update completed.")
