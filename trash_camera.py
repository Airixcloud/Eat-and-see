import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import tensorflow as tf
import json

# =========================
# Settings for Trash Classifier
# =========================
IMAGE_SIZE = 224
# Chosen on the 1500 real_world test images: below 60% the model is wrong more often,
# warning here catches ~62% of wrong answers while flagging ~16% of correct ones
LOW_CONFIDENCE_THRESHOLD = 0.60

def compute_iou(boxA, boxB):
    """Tính Intersection over Union giữa 2 bounding box (x1, y1, x2, y2)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class TrashClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trash Classifier Camera - Khóa mục tiêu & Phân loại rác")
        
        # Tự động căn chỉnh kích thước cửa sổ để vừa vặn với màn hình (kể cả màn hình 1366x768)
        win_w, win_h = 960, 680
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = 10
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.configure(bg="#212121")

        # --- LOAD MODELS ---
        print("Đang tải mô hình YOLOv8...")
        self.yolo_model = YOLO("yolov8n.pt")
        
        print("Đang tải mô hình Trash Classifier...")
        self.classifier_model = tf.keras.models.load_model("trash_classifier.keras")
        
        with open("class_names.json", "r") as f:
            self.class_names = json.load(f)

        self.category_map = {
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

        # Danh mục thiết bị điện tử từ YOLOv8 (không thuộc 30 loại rác gia dụng thông thường)
        self.ewaste_classes = {
            "cell phone": "Điện Thoại Di Động",
            "laptop": "Máy Tính Xách Tay",
            "mouse": "Chuột Máy Tính",
            "keyboard": "Bàn Phím Máy Tính",
            "tv": "Màn Hình / TV",
            "remote": "Điều Khiển Từ Xa",
            "microwave": "Lò Vi Sóng",
            "toaster": "Máy Nướng Bánh Mì",
        }

        self.cap = None
        self.mode = "CAMERA"  # "CAMERA" hoặc "STATIC_IMAGE"
        self.detected_objects = []
        
        # Biến khóa mục tiêu ổn định
        self.locked_box = None
        self.locked_name = ""
        self.locked_conf = 0.0
        self.is_manual_lock = False
        self.lock_start_time = None
        self.has_captured = False
        self.roi_to_save = None

        self.current_frame = None
        self.loaded_image = None
        self.thumb_photo = None

        # Biến vẽ khung thủ công bằng chuột
        self.is_drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_mouse_x = 0
        self.current_mouse_y = 0

        # --- GIAO DIỆN GUI (Khung ảnh mở rộng to hơn để quét dễ dàng) ---
        self.canvas_w = 680
        self.canvas_h = 440
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="black", highlightthickness=0)
        self.canvas.pack(pady=3)
        
        # Gắn sự kiện click & kéo chuột vẽ khung
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.status_label = tk.Label(
            root, 
            text="[HƯỚNG DẪN] Mở Camera -> Click khung hoặc KÉO CHUỘT VẼ KHUNG quanh rác để phân loại!", 
            font=("Arial", 10, "bold"), 
            fg="#00E676",
            bg="#212121"
        )
        self.status_label.pack(pady=2)

        # Hàng nút điều khiển ĐƯỢC ĐẶT NGAY DƯỚI CANVAS (Luôn hiển thị rõ ràng)
        btn_frame = tk.Frame(root, bg="#212121")
        btn_frame.pack(pady=4)

        self.btn_toggle = tk.Button(
            btn_frame, 
            text="🎥 Mở Camera", 
            font=("Arial", 10, "bold"), 
            bg="#4CAF50", 
            fg="white", 
            padx=10, 
            pady=4, 
            command=self.toggle_camera
        )
        self.btn_toggle.grid(row=0, column=0, padx=5)

        self.btn_center = tk.Button(
            btn_frame, 
            text="🎯 Khóa Vùng Giữa", 
            font=("Arial", 10, "bold"), 
            bg="#009688", 
            fg="white", 
            padx=10, 
            pady=4, 
            command=self.lock_center_area
        )
        self.btn_center.grid(row=0, column=1, padx=5)

        self.btn_load = tk.Button(
            btn_frame, 
            text="📁 Tải Ảnh Từ Thiết Bị", 
            font=("Arial", 10, "bold"), 
            bg="#2196F3", 
            fg="white", 
            padx=10, 
            pady=4, 
            command=self.load_image_from_device
        )
        self.btn_load.grid(row=0, column=2, padx=5)

        self.btn_unlock = tk.Button(
            btn_frame, 
            text="🔓 Hủy Khóa / Làm Mới", 
            font=("Arial", 10, "bold"), 
            bg="#FF9800", 
            fg="white", 
            padx=10, 
            pady=4, 
            command=self.unlock_target
        )
        self.btn_unlock.grid(row=0, column=3, padx=5)

        self.btn_save = tk.Button(
            btn_frame, 
            text="� Lưu Ảnh Đồ Vật", 
            font=("Arial", 10, "bold"), 
            bg="#9C27B0", 
            fg="white", 
            padx=10, 
            pady=4, 
            command=self.save_captured_image
        )
        self.btn_save.grid(row=0, column=4, padx=5)

        # Frame hiển thị kết quả phân loại
        self.result_frame = tk.Frame(root, bg="#2d2d2d", padx=12, pady=6, relief=tk.RIDGE, borderwidth=2)
        self.result_frame.pack(pady=4, fill=tk.X, padx=30)

        # Cột ảnh thumbnail đã chụp
        self.thumb_label = tk.Label(self.result_frame, bg="#2d2d2d", width=60, height=60)
        self.thumb_label.grid(row=0, column=0, rowspan=3, padx=(0, 12))

        info_frame = tk.Frame(self.result_frame, bg="#2d2d2d")
        info_frame.grid(row=0, column=1, sticky="w")

        self.res_item_label = tk.Label(info_frame, text="Vật thể: ---", font=("Arial", 11, "bold"), fg="white", bg="#2d2d2d")
        self.res_item_label.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.res_conf_label = tk.Label(info_frame, text="Độ tự tin: ---", font=("Arial", 10), fg="#B0BEC5", bg="#2d2d2d")
        self.res_conf_label.grid(row=0, column=1, sticky="w", padx=(0, 15))

        self.res_cat_label = tk.Label(info_frame, text="Loại rác: ---", font=("Arial", 11, "bold"), fg="#FFEB3B", bg="#2d2d2d")
        self.res_cat_label.grid(row=0, column=2, sticky="w")

        self.res_top3_label = tk.Label(self.result_frame, text="Top 3 dự đoán: ---", font=("Arial", 9), fg="#90CAF9", bg="#2d2d2d")
        self.res_top3_label.grid(row=1, column=1, sticky="w", pady=(2, 0))

        self.res_hint_label = tk.Label(self.result_frame, text="", font=("Arial", 8, "italic"), fg="#FFB74D", bg="#2d2d2d")
        self.res_hint_label.grid(row=2, column=1, sticky="w")

    def lock_center_area(self):
        """Khóa vùng trung tâm của camera để phân loại vật thể bất kỳ (kể cả khi YOLO không nhận diện)"""
        if self.mode != "CAMERA" or self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("Thông báo", "Vui lòng mở camera trước khi khóa vùng trung tâm!")
            return

        cx1 = int(self.canvas_w * 0.20)
        cy1 = int(self.canvas_h * 0.15)
        cx2 = int(self.canvas_w * 0.80)
        cy2 = int(self.canvas_h * 0.85)
        
        self.locked_box = (cx1, cy1, cx2, cy2)
        self.locked_name = "Vùng Trung Tâm"
        self.locked_conf = 1.0
        self.is_manual_lock = True
        self.lock_start_time = time.time()
        self.has_captured = False
        self.roi_to_save = None
        self.status_label.config(text="🎯 ĐÃ KHÓA VÙNG TRUNG TÂM! Giữ yên đồ vật trong 2s...", fg="#FFEB3B")

    def toggle_camera(self):
        if self.cap is None:
            self.loaded_image = None
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể kết nối với Webcam!")
                self.cap = None
                return

            self.mode = "CAMERA"
            self.btn_toggle.config(text="🛑 Tắt Camera", bg="#f44336")
            self.unlock_target()
            self.update_feed()
        else:
            self.stop_camera()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.btn_toggle.config(text="🎥 Mở Camera", bg="#4CAF50")
        if self.mode == "CAMERA":
            self.status_label.config(text="[OFF] CHƯA BẬT CAMERA", fg="gray")
            self.canvas.delete("all")

    def load_image_from_device(self):
        """Cho phép tải file ảnh từ máy tính"""
        self.stop_camera()
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh từ thiết bị",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )

        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.loaded_image = cv2.resize(img, (self.canvas_w, self.canvas_h))
                self.mode = "STATIC_IMAGE"
                self.unlock_target()
                self.process_static_image()
            else:
                messagebox.showerror("Lỗi", "Không thể đọc file ảnh này!")

    def process_static_image(self):
        """Phát hiện vật thể bằng YOLO trên ảnh tĩnh"""
        if self.loaded_image is None:
            return

        frame = self.loaded_image.copy()
        results = self.yolo_model(frame, stream=True, verbose=False)
        
        new_detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # Bỏ qua con người
                    continue
                conf = float(box.conf[0])
                if conf > 0.30:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = self.yolo_model.names.get(cls_id, f"obj_{cls_id}")
                    new_detected_objects.append({
                        "box": (x1, y1, x2, y2),
                        "cls_id": cls_id,
                        "name": name,
                        "conf": conf
                    })

        self.detected_objects = new_detected_objects
        if self.detected_objects:
            self.status_label.config(
                text=f"📁 TẢI ẢNH XONG: Tìm thấy {len(self.detected_objects)} vật thể! Click vào khung hoặc kéo chuột vẽ khung quanh đồ vật.", 
                fg="#00E676"
            )
        else:
            self.status_label.config(
                text="📁 TẢI ẢNH XONG: Chưa nhận dạng được khung tự động. Bạn hãy KÉO CHUỘT VẼ KHUNG quanh vật thể!", 
                fg="#FFEB3B"
            )
        self.display_static_image()

    def display_static_image(self):
        if self.loaded_image is None:
            return

        display_img = self.loaded_image.copy()

        # Vẽ các khung phát hiện tự động
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj["box"]
            name = obj["name"]
            conf = obj["conf"]
            if self.locked_box is not None and compute_iou(self.locked_box, (x1, y1, x2, y2)) > 0.5:
                continue
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(display_img, f"{name} {int(conf*100)}%", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Nếu có khung đang khóa
        if self.locked_box is not None:
            lx1, ly1, lx2, ly2 = self.locked_box
            cv2.rectangle(display_img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            cv2.putText(display_img, f"LOCKED: {self.locked_name}", (lx1, max(15, ly1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Nếu đang kéo chuột vẽ khung
        if self.is_drawing:
            cv2.rectangle(display_img, (self.start_x, self.start_y), (self.current_mouse_x, self.current_mouse_y), (0, 140, 255), 2)

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk

    def unlock_target(self):
        self.locked_box = None
        self.locked_name = ""
        self.locked_conf = 0.0
        self.is_manual_lock = False
        self.lock_start_time = None
        self.has_captured = False
        self.roi_to_save = None
        self.status_label.config(text="[READY] Click chọn khung YOLO hoặc KÉO CHUỘT VẼ KHUNG quanh rác!", fg="#00E676")
        self.res_item_label.config(text="Vật thể: ---")
        self.res_conf_label.config(text="Độ tự tin: ---")
        self.res_cat_label.config(text="Loại rác: ---")
        self.res_top3_label.config(text="Top 3 dự đoán: ---")
        self.res_hint_label.config(text="")
        self.thumb_label.config(image="")
        self.thumb_photo = None
        if self.mode == "STATIC_IMAGE":
            self.display_static_image()

    def crop_with_margin(self, frame, box, margin_ratio=0.08):
        """Cắt ảnh ROI có thêm viền đệm ngữ cảnh xung quanh để tăng độ chính xác phân loại"""
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        margin_x = int(bw * margin_ratio)
        margin_y = int(bh * margin_ratio)
        
        cy1 = max(0, y1 - margin_y)
        cy2 = min(frame.shape[0], y2 + margin_y)
        cx1 = max(0, x1 - margin_x)
        cx2 = min(frame.shape[1], x2 + margin_x)
        
        if cy2 <= cy1 or cx2 <= cx1:
            return frame[y1:y2, x1:x2].copy()
        return frame[cy1:cy2, cx1:cx2].copy()

    def classify_trash(self, roi):
        """Hàm phân loại rác từ vùng ảnh cắt được (ROI)"""
        try:
            if roi is None or roi.size == 0:
                return "Lỗi", 0.0, "unknown", []

            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_float = img.astype(np.float32)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_float)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.classifier_model.predict(img_array, verbose=0)
            preds = predictions[0]
            top_pred_idx = int(np.argmax(preds))

            top5_indices = np.argsort(preds)[-5:][::-1]
            top3_results = [(self.class_names[idx], float(preds[idx])) for idx in top5_indices[:3]]
            
            confidence = float(preds[top_pred_idx])
            item = self.class_names[top_pred_idx]
            category = self.category_map.get(item, "unknown")
            
            return item, confidence, category, top3_results
        except Exception as e:
            print(f"Lỗi phân loại: {e}")
            return "Lỗi", 0.0, "unknown", []

    def update_result_ui(self, item, conf, cat, top3=None):
        self.res_item_label.config(text=f"Vật thể: {item.replace('_', ' ').title()}", fg="white")
        self.res_conf_label.config(text=f"Độ tự tin: {conf*100:.1f}%", fg="#B0BEC5")
        self.res_cat_label.config(text=f"Loại rác: {cat.upper()}")
        
        if cat == "recyclable":
            self.res_cat_label.config(fg="#00E676")
        elif cat == "organic":
            self.res_cat_label.config(fg="#8BC34A")
        elif "e-waste" in cat:
            self.res_cat_label.config(fg="#FF5722")
        else:
            self.res_cat_label.config(fg="#F44336")

        if "e-waste" in cat:
            self.res_top3_label.config(text=f"Nhận diện bởi: YOLOv8 Object Detection ({conf*100:.1f}%)")
            self.res_hint_label.config(
                text="ℹ️ Đây là thiết bị điện tử (E-Waste). Cần đưa đến điểm thu gom rác điện tử, không vứt chung rác sinh hoạt!", 
                fg="#FF8A80"
            )
        elif top3 and len(top3) > 1:
            top_str = "Top 3 dự đoán: " + " | ".join([f"{name.replace('_', ' ').title()} ({prob*100:.1f}%)" for name, prob in top3])
            self.res_top3_label.config(text=top_str)
            print(top_str)

            # Cảnh báo nếu độ tự tin thấp
            if conf < LOW_CONFIDENCE_THRESHOLD:
                self.res_item_label.config(text=f"Vật thể: {item.replace('_', ' ').title()} (chưa chắc chắn)", fg="#FFB74D")
                self.res_conf_label.config(fg="#FFB74D")
                self.res_hint_label.config(
                    text=f"⚠️ Độ tự tin dưới {LOW_CONFIDENCE_THRESHOLD*100:.0f}%: kết quả có thể sai, hãy xem Top 3. Thử lại: cách camera 25-40cm, đủ sáng, lấy trọn vật thể.",
                    fg="#FFB74D"
                )
            else:
                self.res_hint_label.config(text="")

        # Cập nhật thumbnail vùng ảnh đã phân loại
        if self.roi_to_save is not None and self.roi_to_save.size > 0:
            try:
                thumb_rgb = cv2.cvtColor(self.roi_to_save, cv2.COLOR_BGR2RGB)
                thumb_pil = Image.fromarray(thumb_rgb)
                thumb_pil.thumbnail((70, 70))
                self.thumb_photo = ImageTk.PhotoImage(thumb_pil)
                self.thumb_label.config(image=self.thumb_photo)
            except Exception as e:
                print(f"Lỗi hiển thị thumbnail: {e}")

    def save_captured_image(self):
        """Lưu ảnh vật thể đã phân loại xuống máy"""
        if self.roi_to_save is None or self.roi_to_save.size == 0:
            messagebox.showwarning("Cảnh báo", "Chưa chọn hoặc chưa phân loại vật thể nào để lưu!")
            return

        default_filename = f"trash_object_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = filedialog.asksaveasfilename(
            title="Lưu ảnh rác vào thiết bị",
            initialfile=default_filename,
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")]
        )

        if file_path:
            cv2.imwrite(file_path, self.roi_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            messagebox.showinfo("Thành công", f"Đã lưu ảnh rác vào thiết bị tại:\n{file_path}")
            self.status_label.config(text=f"✅ ĐÃ LƯU XONG: {os.path.basename(file_path)}", fg="#00E676")

    # --- SỰ KIỆN CHUỘT VẼ KHUNG / CLICK KHÓA MỤC TIÊU ---
    def on_mouse_down(self, event):
        if self.mode == "CAMERA" and (self.cap is None or not self.cap.isOpened()):
            return
        if self.mode == "STATIC_IMAGE" and self.loaded_image is None:
            return

        self.is_drawing = True
        self.start_x, self.start_y = event.x, event.y
        self.current_mouse_x, self.current_mouse_y = event.x, event.y

    def on_mouse_drag(self, event):
        if self.is_drawing:
            self.current_mouse_x = max(0, min(event.x, self.canvas_w))
            self.current_mouse_y = max(0, min(event.y, self.canvas_h))
            if self.mode == "STATIC_IMAGE":
                self.display_static_image()

    def on_mouse_up(self, event):
        if not self.is_drawing:
            return
        self.is_drawing = False
        end_x = max(0, min(event.x, self.canvas_w))
        end_y = max(0, min(event.y, self.canvas_h))

        dx = abs(end_x - self.start_x)
        dy = abs(end_y - self.start_y)

        # Nếu kéo chuột tạo khung rộng (> 20px) -> Khóa khung thủ công
        if dx > 20 and dy > 20:
            x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
            x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
            
            self.locked_box = (x1, y1, x2, y2)
            self.locked_name = "Khung Tự Chọn"
            self.locked_conf = 1.0
            self.is_manual_lock = True
            self.lock_start_time = time.time()
            self.has_captured = False

            if self.mode == "STATIC_IMAGE" and self.loaded_image is not None:
                self.roi_to_save = self.crop_with_margin(self.loaded_image, self.locked_box)
                self.has_captured = True
                item, conf, cat, top3 = self.classify_trash(self.roi_to_save)
                self.update_result_ui(item, conf, cat, top3)
                self.display_static_image()
            else:
                self.status_label.config(text="🎯 ĐÃ KHÓA KHUNG TỰ VẼ! Giữ yên camera...", fg="#FFEB3B")
        else:
            # Click chuột -> Chọn khung YOLO đã phát hiện
            click_x, click_y = event.x, event.y
            if self.detected_objects:
                selected_obj = None
                min_area = float('inf')

                for obj in self.detected_objects:
                    x1, y1, x2, y2 = obj["box"]
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < min_area:
                            min_area = area
                            selected_obj = obj

                if selected_obj is not None:
                    self.locked_box = selected_obj["box"]
                    self.locked_name = selected_obj["name"]
                    self.locked_conf = selected_obj["conf"]
                    self.is_manual_lock = False
                    self.lock_start_time = time.time()
                    self.has_captured = False
                    self.roi_to_save = None

                    if self.mode == "STATIC_IMAGE" and self.loaded_image is not None:
                        self.roi_to_save = self.crop_with_margin(self.loaded_image, self.locked_box)
                        self.has_captured = True
                        if not self.is_manual_lock and self.locked_name in self.ewaste_classes:
                            item = self.ewaste_classes[self.locked_name]
                            conf = self.locked_conf
                            cat = "e-waste"
                            top3 = [(item, conf)]
                        else:
                            item, conf, cat, top3 = self.classify_trash(self.roi_to_save)
                        self.update_result_ui(item, conf, cat, top3)
                        self.display_static_image()

    def update_feed(self):
        if self.mode != "CAMERA" or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.canvas_w, self.canvas_h))
            self.current_frame = frame.copy()
            
            results = self.yolo_model(frame, stream=True, verbose=False)
            
            new_detected_objects = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # Bỏ qua con người
                        continue

                    conf = float(box.conf[0])
                    if conf > 0.30:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        name = self.yolo_model.names.get(cls_id, f"obj_{cls_id}")
                        new_detected_objects.append({
                            "box": (x1, y1, x2, y2),
                            "cls_id": cls_id,
                            "name": name,
                            "conf": conf
                        })

            self.detected_objects = new_detected_objects

            # Theo dõi mục tiêu đã khóa qua các khung hình bằng IoU
            if self.locked_box is not None and not self.is_manual_lock:
                best_match = None
                best_iou = 0.0
                for obj in self.detected_objects:
                    iou = compute_iou(self.locked_box, obj["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj

                # Nếu tìm thấy vật thể tương ứng ở vị trí lân cận -> Cập nhật vị trí bám theo
                if best_match is not None and best_iou > 0.20:
                    self.locked_box = best_match["box"]
                    self.locked_name = best_match["name"]
                    self.locked_conf = best_match["conf"]
                # Nếu không tìm thấy trong 1 frame ngắn, giữ nguyên locked_box trước đó (không để mất khóa)

            # Đếm ngược & phân loại khi khóa vật thể trên camera
            if self.locked_box is not None and not self.has_captured:
                elapsed = time.time() - self.lock_start_time
                remaining = max(0, 2 - int(elapsed))

                if remaining > 0:
                    self.status_label.config(
                        text=f"🎯 ĐÃ KHÓA [{self.locked_name.upper()}]! ĐANG PHÂN TÍCH TRONG: {remaining}s...", 
                        fg="#FFEB3B"
                    )
                else:
                    self.has_captured = True
                    self.roi_to_save = self.crop_with_margin(frame, self.locked_box)
                    
                    # Nếu YOLO phát hiện là thiết bị điện tử (e-waste)
                    if not self.is_manual_lock and self.locked_name in self.ewaste_classes:
                        item = self.ewaste_classes[self.locked_name]
                        conf = self.locked_conf
                        cat = "e-waste"
                        top3 = [(item, conf)]
                    else:
                        item, conf, cat, top3 = self.classify_trash(self.roi_to_save)

                    self.update_result_ui(item, conf, cat, top3)
                    self.status_label.config(text=f"✅ ĐÃ PHÂN LOẠI XONG [{item.replace('_', ' ').title()}]!", fg="#00E676")

            # Chuẩn bị khung hình hiển thị
            display_frame = frame.copy()

            # Vẽ các khung phát hiện tự động
            for obj in self.detected_objects:
                x1, y1, x2, y2 = obj["box"]
                name = obj["name"]
                conf = obj["conf"]
                # Không vẽ đè nếu trùng với locked_box
                if self.locked_box is not None and compute_iou(self.locked_box, (x1, y1, x2, y2)) > 0.4:
                    continue
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(display_frame, f"{name} {int(conf*100)}%", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Vẽ khung đang khóa
            if self.locked_box is not None:
                lx1, ly1, lx2, ly2 = self.locked_box
                color = (0, 255, 0) if self.has_captured else (0, 230, 255)
                cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), color, 2)
                lock_text = f"LOCKED: {self.locked_name}"
                cv2.putText(display_frame, lock_text, (lx1, max(15, ly1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Hiển thị vẽ khung drag trên camera
            if self.is_drawing:
                cv2.rectangle(display_frame, (self.start_x, self.start_y), (self.current_mouse_x, self.current_mouse_y), (0, 140, 255), 2)

            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.img_tk = img_tk

        self.root.after(30, self.update_feed)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrashClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()
