import io
import json
import os
import sys
import numpy as np
from PIL import Image

# Turn off verbose tensorflow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.55

app = FastAPI(title="Greener - Intelligent Multi-Stage Waste Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Friendly Vietnamese translations & advice
CATEGORY_INFO = {
    "recyclable": {
        "title": "Rác Tái Chế",
        "color": "#10B981", # Emerald green
        "badge": "badge-recyclable",
        "action": "Thu gom riêng, làm sạch chất bẩn/chất lỏng còn sót và đưa vào thùng rác tái chế màu xanh lá / trắng."
    },
    "organic": {
        "title": "Rác Hữu Cơ",
        "color": "#F59E0B", # Amber / Orange
        "badge": "badge-organic",
        "action": "Có thể ủ làm phân bón hữu cơ hoặc bỏ vào thùng rác hữu cơ/thức ăn thừa để xử lý vi sinh."
    },
    "non-recyclable": {
        "title": "Rác Vô Cơ / Không Thể Tái Chế",
        "color": "#EF4444", # Red
        "badge": "badge-non-recyclable",
        "action": "Bỏ vào thùng rác vô cơ hoặc rác sinh hoạt thông thường để chôn lấp/thiêu hủy tiêu chuẩn."
    },
    "hazardous": {
        "title": "Rác Điện Tử / Nguy Hại",
        "color": "#8B5CF6", # Purple
        "badge": "badge-hazardous",
        "action": "Chứa linh kiện, kim loại nặng hoặc vi mạch độc hại. Tuyệt đối KHÔNG vứt cùng rác thông thường. Hãy nộp tại các điểm thu gom rác điện tử chuyên trách."
    }
}

VI_NAMES = {
    "aerosol_cans": "Bình xịt khí nén",
    "aluminum_food_cans": "Hộp đồ ăn bằng nhôm",
    "aluminum_soda_cans": "Lon nước ngọt / bia nhôm",
    "cardboard_boxes": "Thùng carton / Hộp giấy",
    "cardboard_packaging": "Bao bì bìa cứng",
    "clothing": "Quần áo cũ / Vải vụn",
    "coffee_grounds": "Bã cà phê",
    "disposable_plastic_cutlery": "Dao muỗng nĩa nhựa dùng 1 lần",
    "eggshells": "Vỏ trứng",
    "food_waste": "Thức ăn thừa",
    "glass_beverage_bottles": "Chai thủy tinh đựng đồ uống",
    "glass_cosmetic_containers": "Lọ mỹ phẩm thủy tinh",
    "glass_food_jars": "Hũ thủy tinh đựng thực phẩm",
    "magazines": "Tạp chí in",
    "newspaper": "Báo giấy",
    "office_paper": "Giấy văn phòng / Giấy A4",
    "paper_cups": "Ly giấy dùng 1 lần",
    "plastic_cup_lids": "Nắp cốc nhựa",
    "plastic_detergent_bottles": "Chai nước giặt / tẩy rửa",
    "plastic_food_containers": "Hộp nhựa đựng thực phẩm",
    "plastic_shopping_bags": "Túi nilon đi chợ / mua sắm",
    "plastic_soda_bottles": "Chai nhựa nước ngọt",
    "plastic_straws": "Ống hút nhựa",
    "plastic_trash_bags": "Túi rác nilon",
    "plastic_water_bottles": "Chai nhựa đựng nước suối",
    "shoes": "Giày dép cũ",
    "steel_food_cans": "Hộp thiếc / đồ hộp sắt",
    "styrofoam_cups": "Ly xốp xì-po",
    "styrofoam_food_containers": "Hộp xốp đựng cơm / thức ăn",
    "tea_bags": "Túi lọc trà",
    # E-waste from YOLO
    "cell phone": "Điện thoại di động",
    "laptop": "Máy tính xách tay",
    "mouse": "Chuột máy tính",
    "keyboard": "Bàn phím máy tính",
    "tv": "Màn hình tivi / Monitor",
    "remote": "Điều khiển từ xa (Remote)",
    "microwave": "Lò vi sóng",
    "toaster": "Máy nướng bánh mì"
}

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

EWASTE_CLASSES = {
    "cell phone": "Điện thoại di động",
    "laptop": "Máy tính xách tay",
    "mouse": "Chuột máy tính",
    "keyboard": "Bàn phím máy tính",
    "tv": "Màn hình / TV",
    "remote": "Điều khiển từ xa",
    "microwave": "Lò vi sóng",
    "toaster": "Máy nướng bánh mì",
}

# Strict Background Exclusion Blacklist
IGNORE_COCO_CLASSES = {
    "person", "chair", "couch", "sofa", "bed", "dining table", "tv", 
    "refrigerator", "oven", "sink", "toilet", "door", "window",
    "bench", "potted plant", "clock", "vase", "wall", "floor"
}

# Food items from COCO that correlate with Organic waste
COCO_FOOD_CLASSES = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake"
}

print(">> Loading Classifier Model...")
classifier_path = os.path.join(BASE_DIR, "trash_classifier.keras")
classifier_model = tf.keras.models.load_model(classifier_path)

with open(os.path.join(BASE_DIR, "class_names.json"), "r", encoding="utf-8") as f:
    class_names = json.load(f)

print(">> Loading YOLOv8 Model...")
yolo_path = os.path.join(BASE_DIR, "yolov8n.pt")
yolo_model = YOLO(yolo_path)
print(">> All Models loaded successfully with High-Accuracy Engine!")


def letterbox_image(pil_img: Image.Image, target_size: int = 224) -> np.ndarray:
    """
    Resizes image strictly preserving aspect ratio and pads borders with neutral gray (128,128,128).
    Prevents squashing/stretching of bottles, cans, boxes!
    """
    w, h = pil_img.size
    scale = target_size / max(w, h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    
    resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    padded.paste(resized, ((target_size - nw) // 2, (target_size - nh) // 2))
    
    arr = tf.keras.preprocessing.image.img_to_array(padded)
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def crop_with_margin(pil_img: Image.Image, box: list, margin_ratio: float = 0.12) -> Image.Image:
    """Adds context padding margin around bounding box to prevent cutting off object features."""
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    
    if cx2 <= cx1 or cy2 <= cy1:
        return pil_img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return pil_img.crop((cx1, cy1, cx2, cy2))


def classify_with_tta(pil_img: Image.Image, coco_hint: str = None):
    """
    Multi-Scale Test-Time Augmentation (TTA) Ensemble Classifier:
    1. Standard Letterbox View (preserving aspect ratio)
    2. Center-Focus Zoom View (85% central crop)
    3. Horizontal Mirror View
    Computes ensemble Softmax distribution and category aggregations.
    """
    w, h = pil_img.size
    
    # View 1: Letterbox
    v1 = letterbox_image(pil_img, IMAGE_SIZE)
    
    # View 2: Center Zoom
    crop_w, crop_h = int(w * 0.85), int(h * 0.85)
    cx, cy = w // 2, h // 2
    z_img = pil_img.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2))
    v2 = letterbox_image(z_img, IMAGE_SIZE)
    
    # View 3: Horizontal Flip
    f_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    v3 = letterbox_image(f_img, IMAGE_SIZE)
    
    batch = np.stack([v1, v2, v3], axis=0)
    preds = classifier_model.predict(batch, verbose=0)
    score = np.mean(preds, axis=0)
    
    # Hybrid Synergy: if YOLO saw a bottle, cup, or food item
    if coco_hint == "bottle":
        bottle_classes = {"plastic_water_bottles", "plastic_soda_bottles", "glass_beverage_bottles", "plastic_detergent_bottles"}
        for idx, c in enumerate(class_names):
            if c in bottle_classes:
                score[idx] *= 1.45
        score = score / np.sum(score)
    elif coco_hint in ("cup", "wine glass"):
        cup_classes = {"paper_cups", "styrofoam_cups", "plastic_cup_lids", "glass_beverage_bottles"}
        for idx, c in enumerate(class_names):
            if c in cup_classes:
                score[idx] *= 1.45
        score = score / np.sum(score)
    elif coco_hint in COCO_FOOD_CLASSES:
        for idx, c in enumerate(class_names):
            if c == "food_waste":
                score[idx] *= 1.8
        score = score / np.sum(score)
    elif coco_hint in ("fork", "knife", "spoon"):
        for idx, c in enumerate(class_names):
            if c == "disposable_plastic_cutlery":
                score[idx] *= 1.8
        score = score / np.sum(score)

    top_idx = int(np.argmax(score))
    class_name = class_names[top_idx]
    confidence = float(np.max(score))

    # Top 3 predictions
    top_indices = np.argsort(score)[::-1][:3]
    top_predictions = [
        {
            "class_name": class_names[i],
            "display_name": VI_NAMES.get(class_names[i], class_names[i]),
            "category": CATEGORY_MAP.get(class_names[i], "non-recyclable"),
            "category_info": CATEGORY_INFO.get(CATEGORY_MAP.get(class_names[i], "non-recyclable")),
            "confidence": round(float(score[i]) * 100, 2)
        }
        for i in top_indices
    ]

    category = CATEGORY_MAP.get(class_name, "non-recyclable")
    cat_info = CATEGORY_INFO.get(category, CATEGORY_INFO["non-recyclable"])

    return {
        "class_name": class_name,
        "display_name": VI_NAMES.get(class_name, class_name),
        "confidence": round(confidence * 100, 2),
        "is_low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        "category": category,
        "category_title": cat_info["title"],
        "category_color": cat_info["color"],
        "category_action": cat_info["action"],
        "top_predictions": top_predictions
    }


@app.post("/api/classify")
async def api_classify(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classify_with_tta(pil_img)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect_and_classify")
async def api_detect_and_classify(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size
        img_area = float(w * h)

        # 1. Run YOLO with confidence threshold 0.30
        yolo_results = yolo_model(pil_img, conf=0.30, verbose=False)
        detected_objects = []

        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                obj_name = r.names[cls_idx].lower()

                # TẦNG 1: Loại bỏ tuyệt đối các class background / người / nội thất
                if obj_name in IGNORE_COCO_CLASSES:
                    continue

                bw = x2 - x1
                bh = y2 - y1
                box_area = float(bw * bh)

                # Bỏ qua box quá nhỏ (< 1.5% diện tích) hoặc quá lớn (> 88% diện tích)
                if box_area < (0.015 * img_area) or box_area > (0.88 * img_area):
                    continue

                # Tính khoảng cách từ tâm box tới tâm màn hình (Center-Priority)
                box_cx = (x1 + x2) / (2.0 * w)
                box_cy = (y1 + y2) / (2.0 * h)
                dist_center = np.sqrt((box_cx - 0.5) ** 2 + (box_cy - 0.5) ** 2)

                # Trọng số ưu tiên vật thể ở trung tâm
                center_score = conf * (1.0 - 0.45 * dist_center)

                # Trường hợp A: Rác điện tử (E-Waste)
                if obj_name in EWASTE_CLASSES:
                    detected_objects.append({
                        "box": [x1, y1, x2, y2],
                        "normalized_box": [round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)],
                        "class_name": obj_name,
                        "display_name": EWASTE_CLASSES[obj_name],
                        "confidence": round(conf * 100, 2),
                        "category": "hazardous",
                        "category_title": CATEGORY_INFO["hazardous"]["title"],
                        "category_color": CATEGORY_INFO["hazardous"]["color"],
                        "category_action": CATEGORY_INFO["hazardous"]["action"],
                        "is_ewaste": True,
                        "priority": center_score
                    })
                else:
                    # Trường hợp B: Cắt vùng ảnh ROI có viền đệm ngữ cảnh + TTA Ensemble
                    roi = crop_with_margin(pil_img, [x1, y1, x2, y2], margin_ratio=0.12)
                    clf_res = classify_with_tta(roi, coco_hint=obj_name)
                    
                    detected_objects.append({
                        "box": [x1, y1, x2, y2],
                        "normalized_box": [round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)],
                        "class_name": clf_res["class_name"],
                        "display_name": clf_res["display_name"],
                        "confidence": clf_res["confidence"],
                        "category": clf_res["category"],
                        "category_title": clf_res["category_title"],
                        "category_color": clf_res["category_color"],
                        "category_action": clf_res["category_action"],
                        "is_ewaste": False,
                        "is_low_confidence": clf_res["is_low_confidence"],
                        "top_predictions": clf_res["top_predictions"],
                        "priority": center_score * (clf_res["confidence"] / 100.0)
                    })

        # Sắp xếp các vật thể phát hiện được: ưu tiên cao nhất cho vật thể ở gần tâm và tự tin nhất
        if detected_objects:
            detected_objects.sort(key=lambda item: item.get("priority", 0), reverse=True)

        # 2. Phân tích vùng lấy nét trung tâm (Focal Center ROI)
        # Nếu YOLO không bắt được class COCO (ví dụ vỏ trứng, bã trà, bìa carton, túi nilon)
        # Ta trích xuất 60% vùng giữa camera nơi người dùng hướng ống kính vào
        cx1 = int(w * 0.18)
        cy1 = int(h * 0.16)
        cx2 = int(w * 0.82)
        cy2 = int(h * 0.84)
        center_focal_roi = pil_img.crop((cx1, cy1, cx2, cy2))
        focal_result = classify_with_tta(center_focal_roi)

        # Phân tích toàn cảnh (Whole-image TTA)
        whole_result = classify_with_tta(pil_img)

        # Nếu không có đối tượng hợp lệ nào từ YOLO, tạo box mục tiêu trung tâm từ focal_result
        if not detected_objects:
            detected_objects.append({
                "box": [cx1, cy1, cx2, cy2],
                "normalized_box": [round(cx1/w, 4), round(cy1/h, 4), round(cx2/w, 4), round(cy2/h, 4)],
                "class_name": focal_result["class_name"],
                "display_name": focal_result["display_name"],
                "confidence": focal_result["confidence"],
                "category": focal_result["category"],
                "category_title": focal_result["category_title"],
                "category_color": focal_result["category_color"],
                "category_action": focal_result["category_action"],
                "is_ewaste": False,
                "is_low_confidence": focal_result["is_low_confidence"],
                "top_predictions": focal_result["top_predictions"],
                "priority": 1.0
            })

        return JSONResponse({
            "image_size": {"width": w, "height": h},
            "whole_image": focal_result if focal_result["confidence"] > whole_result["confidence"] else whole_result,
            "detected_objects": detected_objects
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


CONSOLIDATED_GROUPS = [
    # ---------------- 1. RÁC TÁI CHẾ (RECYCLABLE) ----------------
    {
        "id": "plastic_bottles",
        "name": "Chai & Bình Nhựa",
        "sub_items": "Chai nước suối, chai nước ngọt, chai nước giặt / tẩy rửa",
        "category": "recyclable",
        "icon": "🧴",
        "action": "Trút sạch chất lỏng, xúc sạch cát bẩn, tháo nắp hoặc bóp dẹp để tiết kiệm diện tích. Cho vào thùng rác tái chế màu xanh lá.",
        "images": [
            "/static/samples/plastic_bottles_1.jpg",
            "/static/samples/plastic_bottles_2.jpg",
            "/static/test_bottle.jpg"
        ]
    },
    {
        "id": "metal_cans",
        "name": "Lon & Hộp Kim Loại",
        "sub_items": "Lon nước ngọt / bia nhôm, đồ hộp kim loại, hộp thiếc, bình xịt",
        "category": "recyclable",
        "icon": "🥫",
        "action": "Rửa sạch dầu mỡ, cặn nước sốt còn đọng bên trong, làm bẹp hoặc gập mép nắp lon để an toàn cho công nhân thu gom rác.",
        "images": [
            "/static/samples/metal_cans_1.jpg",
            "/static/samples/metal_cans_2.jpg"
        ]
    },
    {
        "id": "cardboard",
        "name": "Thùng Giấy & Bìa Carton",
        "sub_items": "Thùng carton giao hàng, bao bì bìa cứng, hộp bánh kẹo",
        "category": "recyclable",
        "icon": "📦",
        "action": "Tháo bỏ băng dính dán trên hộp, gỡ ghim bấm kim loại, gấp phẳng hộp và giữ khô ráo tuyệt đối (không bị dính dầu mỡ).",
        "images": [
            "/static/samples/cardboard_1.jpg",
            "/static/samples/cardboard_2.jpg"
        ]
    },
    {
        "id": "paper",
        "name": "Giấy Báo & Sách Vở, Giấy A4",
        "sub_items": "Báo giấy in, tạp chí, giấy văn phòng A4, tài liệu cũ",
        "category": "recyclable",
        "icon": "📰",
        "action": "Thu gom gọn gàng, buộc thành thếp hoặc xếp trong túi giấy, tránh để ẩm ướt hoặc dính chất lỏng thực phẩm.",
        "images": [
            "/static/samples/paper_1.jpg",
            "/static/samples/paper_2.jpg"
        ]
    },
    {
        "id": "glass",
        "name": "Chai Lọ Thủy Tinh",
        "sub_items": "Chai nước ngọt / bia thủy tinh, hũ thực phẩm, lọ mỹ phẩm",
        "category": "recyclable",
        "icon": "🍾",
        "action": "Súc rửa sạch bên trong, để ráo. Nếu chai vỡ cần bọc giấy báo cẩn thận ghi chú bên ngoài để bảo vệ người thu gom.",
        "images": [
            "/static/samples/glass_1.jpg",
            "/static/samples/glass_2.jpg"
        ]
    },
    {
        "id": "clothing_shoes",
        "name": "Quần Áo & Giày Dép Cũ",
        "sub_items": "Quần áo vải cũ, vải vụn, giày dép thể thao đã qua sử dụng",
        "category": "recyclable",
        "icon": "👕",
        "action": "Giặt sạch, đóng gói khô ráo. Có thể tái chế sợi dệt, làm giẻ lau công nghiệp hoặc gửi đến các trạm quyên góp từ thiện.",
        "images": [
            "/static/samples/clothing_shoes_1.jpg",
            "/static/samples/clothing_shoes_2.jpg"
        ]
    },

    # ---------------- 2. RÁC HỮU CƠ (ORGANIC) ----------------
    {
        "id": "food_waste",
        "name": "Thức Ăn Thừa & Rau Củ Quả",
        "sub_items": "Cơm thừa, thức ăn ôi thiu, vỏ rau củ quả, cuống lá, vỏ hạt",
        "category": "organic",
        "icon": "🥗",
        "action": "Chắt ráo nước để hạn chế rỉ dịch hôi thối. Bỏ vào thùng rác hữu cơ hoặc ủ vi sinh làm phân bón hữu cơ dinh dưỡng cho cây.",
        "images": [
            "/static/samples/food_waste_1.jpg",
            "/static/samples/food_waste_2.jpg"
        ]
    },
    {
        "id": "coffee_tea",
        "name": "Bã Cà Phê & Bã Trà",
        "sub_items": "Bã cà phê pha phin/máy, túi lọc trà nhúng, bã trà xanh",
        "category": "organic",
        "icon": "☕",
        "action": "Là nguồn phân bón giàu nitơ cho cây cảnh hoặc dùng khử mùi tủ lạnh, phòng ngủ tự nhiên trước khi bón đất.",
        "images": [
            "/static/samples/coffee_tea_1.jpg",
            "/static/samples/coffee_tea_2.jpg"
        ]
    },
    {
        "id": "eggshells",
        "name": "Vỏ Trứng Gia Cầm",
        "sub_items": "Vỏ trứng gà, trứng vịt, trứng cút",
        "category": "organic",
        "icon": "🥚",
        "action": "Nghiền nhỏ bón gốc cây bổ sung Canxi cực tốt cho đất trồng, hoặc trộn vào thùng ủ phân hữu cơ gia đình.",
        "images": [
            "/static/samples/eggshells_1.jpg",
            "/static/samples/eggshells_2.jpg"
        ]
    },

    # ---------------- 3. RÁC VÔ CƠ / KHÔNG THỂ TÁI CHẾ (NON-RECYCLABLE) ----------------
    {
        "id": "styrofoam",
        "name": "Hộp Xốp & Ly Xốp",
        "sub_items": "Hộp xốp đựng cơm / thức ăn nóng, ly xốp xì-po giữ nhiệt",
        "category": "non-recyclable",
        "icon": "🥡",
        "action": "Xốp EPS dính dầu mỡ không thể tái chế kinh tế. Gom gọn và bỏ vào thùng rác vô cơ sinh hoạt để thiêu hủy tiêu chuẩn.",
        "images": [
            "/static/samples/styrofoam_1.jpg",
            "/static/samples/styrofoam_2.jpg"
        ]
    },
    {
        "id": "single_use_plastic",
        "name": "Đồ Nhựa Dùng 1 Lần & Ly Giấy",
        "sub_items": "Ly giấy tráng màng nhựa PE, ống hút nhựa, dao muỗng nĩa nhựa 1 lần",
        "category": "non-recyclable",
        "icon": "🥤",
        "action": "Vật liệu hỗn hợp khó phân tách tái chế. Hãy bỏ vào thùng rác vô cơ và hạn chế sử dụng đồ nhựa dùng 1 lần trong đời sống.",
        "images": [
            "/static/samples/single_use_plastic_1.jpg",
            "/static/samples/single_use_plastic_2.jpg"
        ]
    },
    {
        "id": "plastic_bags",
        "name": "Túi Nilon & Bao Bì Nhựa Dẻo",
        "sub_items": "Túi nilon đi chợ, túi rác đen, màng bọc thực phẩm dính dầu mỡ",
        "category": "non-recyclable",
        "icon": "🛍️",
        "action": "Buộc kín miệng túi rác, vứt vào thùng rác vô cơ tập trung để xe rác đưa đến bãi xử lý rác thải chôn lấp hợp vệ sinh.",
        "images": [
            "/static/samples/plastic_bags_1.jpg",
            "/static/samples/plastic_bags_2.jpg"
        ]
    },

    # ---------------- 4. RÁC ĐIỆN TỬ & NGUY HẠI (HAZARDOUS) ----------------
    {
        "id": "phone_tablet",
        "name": "Điện Thoại & Máy Tính Bảng",
        "sub_items": "Điện thoại di động cũ, smartphone vỡ màn hình, máy tính bảng",
        "category": "hazardous",
        "icon": "📱",
        "action": "Chứa pin Lithium và kim loại quý. TUYỆT ĐỐI KHÔNG đốt hay vứt thùng rác sinh hoạt. Nộp tại các điểm thu gom rác điện tử (E-waste).",
        "images": [
            "/static/samples/phone_tablet_1.jpg",
            "/static/samples/phone_tablet_2.jpg"
        ]
    },
    {
        "id": "laptop_computer",
        "name": "Máy Tính Xách Tay & PC",
        "sub_items": "Laptop, máy vi tính để bàn, bộ nguồn máy tính cũ",
        "category": "hazardous",
        "icon": "💻",
        "action": "Chứa bo mạch vi xử lý phức tạp. Đem đến các trung tâm bảo hành hoặc điểm thu gom rác công nghệ để bóc tách kim loại an toàn.",
        "images": [
            "/static/samples/laptop_1.jpg",
            "/static/samples/laptop_2.jpg"
        ]
    },
    {
        "id": "mouse_keyboard",
        "name": "Chuột, Bàn Phím & Phụ Kiện",
        "sub_items": "Chuột máy tính, bàn phím cơ/văn phòng, tai nghe, cáp sạc",
        "category": "hazardous",
        "icon": "⌨️",
        "action": "Có mạch điện tử và dây đồng. Gom cùng các thiết bị điện tử hỏng để nộp tại trạm thu gom rác công nghệ chuyên trách.",
        "images": [
            "/static/samples/mouse_keyboard_1.jpg",
            "/static/samples/mouse_keyboard_2.jpg"
        ]
    },
    {
        "id": "tv_appliances",
        "name": "Màn Hình Tivi & Đồ Điện Gia Dụng",
        "sub_items": "Màn hình tivi, điều khiển remote, lò vi sóng, máy nướng hỏng",
        "category": "hazardous",
        "icon": "📺",
        "action": "Chứa thủy ngân trong đèn hình hoặc tụ điện cao áp. Hãy liên hệ đơn vị thu mua ve chai điện tử hoặc điểm tập kết nguy hại phường/xã.",
        "images": [
            "/static/samples/tv_appliances_1.jpg",
            "/static/samples/tv_appliances_2.jpg"
        ]
    }
]


@app.get("/api/classes")
async def get_classes():
    items = []
    for g in CONSOLIDATED_GROUPS:
        cat_info = CATEGORY_INFO.get(g["category"], CATEGORY_INFO["non-recyclable"])
        items.append({
            "id": g["id"],
            "name": g["name"],
            "sub_items": g["sub_items"],
            "category": g["category"],
            "category_title": cat_info["title"],
            "color": cat_info["color"],
            "icon": g["icon"],
            "action": g["action"],
            "images": g["images"]
        })
    return JSONResponse(items)


# Serve Static Frontend Files
web_dir = os.path.join(BASE_DIR, "web")
os.makedirs(web_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(web_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Trang web đang được tải...</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
