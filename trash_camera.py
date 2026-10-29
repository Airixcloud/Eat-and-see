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

class TrashClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trash Classifier Camera - Khóa mục tiêu & Phân loại rác")
        self.root.geometry("900x820")
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

        self.cap = None
        self.mode = "CAMERA"  # "CAMERA" hoặc "STATIC_IMAGE"
        self.detected_objects = []
        self.locked_object_index = None  
        self.lock_start_time = None
        self.has_captured = False
        self.roi_to_save = None

        self.current_frame = None
        self.loaded_image = None

        # Biến vẽ khung thủ công bằng chuột
        self.is_drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_mouse_x = 0
        self.current_mouse_y = 0

        # --- GIAO DIỆN GUI ---
        self.canvas_w = 640
        self.canvas_h = 480
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # Gắn sự kiện click & kéo chuột vẽ khung
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.status_label = tk.Label(
            root, 
            text="[HƯỚNG DẪN] Mở Camera hoặc Tải Ảnh -> Click/Vẽ khung để khóa & phân loại rác!", 
            font=("Arial", 11, "bold"), 
            fg="#00E676",
            bg="#212121"
        )
        self.status_label.pack(pady=5)

        # Frame hiển thị kết quả phân loại
        self.result_frame = tk.Frame(root, bg="#333333", padx=20, pady=10, relief=tk.RIDGE, borderwidth=2)
        self.result_frame.pack(pady=5, fill=tk.X, padx=50)

        self.res_item_label = tk.Label(self.result_frame, text="Vật thể: ---", font=("Arial", 11), fg="white", bg="#333333")
        self.res_item_label.grid(row=0, column=0, sticky="w", padx=10)

        self.res_conf_label = tk.Label(self.result_frame, text="Độ tự tin: ---", font=("Arial", 11), fg="white", bg="#333333")
        self.res_conf_label.grid(row=0, column=1, sticky="w", padx=10)

        self.res_cat_label = tk.Label(self.result_frame, text="Loại rác: ---", font=("Arial", 11, "bold"), fg="#FFEB3B", bg="#333333")
        self.res_cat_label.grid(row=0, column=2, sticky="w", padx=10)

        btn_frame = tk.Frame(root, bg="#212121")
        btn_frame.pack(pady=10)

        self.btn_toggle = tk.Button(
            btn_frame, 
            text="🎥 Mở Camera", 
            font=("Arial", 10, "bold"), 
            bg="#4CAF50", 
            fg="white", 
            padx=10, 
            pady=5, 
            command=self.toggle_camera
        )
        self.btn_toggle.grid(row=0, column=0, padx=6)

        self.btn_load = tk.Button(
            btn_frame, 
            text="📁 Tải Ảnh Từ Thiết Bị", 
            font=("Arial", 10, "bold"), 
            bg="#2196F3", 
            fg="white", 
            padx=10, 
            pady=5, 
            command=self.load_image_from_device
        )
        self.btn_load.grid(row=0, column=1, padx=6)

        self.btn_unlock = tk.Button(
            btn_frame, 
            text="🔓 Hủy Khóa / Làm Mới", 
            font=("Arial", 10, "bold"), 
            bg="#FF9800", 
            fg="white", 
            padx=10, 
            pady=5, 
            command=self.unlock_target
        )
        self.btn_unlock.grid(row=0, column=2, padx=6)

        self.btn_save = tk.Button(
            btn_frame, 
            text="💾 Lưu Ảnh Đồ Vật", 
            font=("Arial", 10, "bold"), 
            bg="#9C27B0", 
            fg="white", 
            padx=10, 
            pady=5, 
            command=self.save_captured_image
        )
        self.btn_save.grid(row=0, column=3, padx=6)

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
                if conf > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    new_detected_objects.append((x1, y1, x2, y2))

        self.detected_objects = new_detected_objects
        self.status_label.config(
            text=f"📁 ĐÃ TẢI ẢNH! Tìm thấy {len(self.detected_objects)} vật thể. Click vào khung hoặc kéo chuột để vẽ!", 
            fg="#00E676"
        )
        self.display_static_image()

    def display_static_image(self):
        if self.loaded_image is None:
            return

        display_img = self.loaded_image.copy()

        # Nếu đang kéo chuột vẽ khung
        if self.is_drawing:
            cv2.rectangle(display_img, (self.start_x, self.start_y), (self.current_mouse_x, self.current_mouse_y), (0, 140, 255), 2)

        # Vẽ các khung phát hiện tự động
        for idx, bbox in enumerate(self.detected_objects):
            x1, y1, x2, y2 = bbox
            if idx == self.locked_object_index:
                color = (0, 255, 0)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_img, "LOCKED", (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (255, 255, 255)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 1)

        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk

    def unlock_target(self):
        self.locked_object_index = None
        self.lock_start_time = None
        self.has_captured = False
        self.roi_to_save = None
        self.status_label.config(text="[READY] Hãy click chọn hoặc kéo chuột vẽ khung quanh rác!", fg="#00E676")
        self.res_item_label.config(text="Vật thể: ---")
        self.res_conf_label.config(text="Độ tự tin: ---")
        self.res_cat_label.config(text="Loại rác: ---")
        if self.mode == "STATIC_IMAGE":
            self.display_static_image()

    def classify_trash(self, roi):
        """Hàm phân loại rác từ vùng ảnh cắt được (ROI)"""
        try:
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_float = img.astype(np.float32)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_float)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.classifier_model.predict(img_array, verbose=0)
            
            # Lấy top 5 lớp có xác suất cao nhất
            top5_indices = np.argsort(predictions[0])[-5:][::-1]
            top3_results = [(self.class_names[idx], float(predictions[0][idx])) for idx in top5_indices[:3]]
            
            predicted_index = top5_indices[0]
            confidence = predictions[0][predicted_index]

            item = self.class_names[predicted_index]
            category = self.category_map.get(item, "unknown")
            
            return item, confidence, category, top3_results
        except Exception as e:
            print(f"Lỗi phân loại: {e}")
            return "Lỗi", 0.0, "unknown", []

    def update_result_ui(self, item, conf, cat, top3=None):
        self.res_item_label.config(text=f"Vật thể: {item.replace('_', ' ').title()}")
        self.res_conf_label.config(text=f"Độ tự tin: {conf*100:.2f}%")
        self.res_cat_label.config(text=f"Loại rác: {cat.upper()}")
        
        if cat == "recyclable":
            self.res_cat_label.config(fg="#00E676")
        elif cat == "organic":
            self.res_cat_label.config(fg="#8BC34A")
        else:
            self.res_cat_label.config(fg="#F44336")

        if top3:
            top3_str = " Top 3 dự đoán: " + ", ".join([f"{name.replace('_', ' ').title()} ({prob*100:.1f}%)" for name, prob in top3])
            print(top3_str)

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
        end_x, end_y = max(0, min(event.x, self.canvas_w)), max(0, min(event.y, self.canvas_h))

        dx = abs(end_x - self.start_x)
        dy = abs(end_y - self.start_y)

        # Nếu kéo chuột tạo khung rộng (> 15px) -> Vẽ khung thủ công
        if dx > 15 and dy > 15:
            x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
            x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
            
            source_frame = self.current_frame if self.mode == "CAMERA" else self.loaded_image
            if source_frame is not None:
                self.roi_to_save = source_frame[y1:y2, x1:x2].copy()
                item, conf, cat, top3 = self.classify_trash(self.roi_to_save)
                self.update_result_ui(item, conf, cat, top3)
                self.status_label.config(text="📸 ĐÃ PHÂN LOẠI KHUNG TỰ VẼ!", fg="#00E676")
        else:
            # Click chuột bình thường -> Chọn khung YOLO đã phát hiện
            click_x, click_y = event.x, event.y
            if self.detected_objects:
                selected_idx = None
                min_area = float('inf')

                for idx, bbox in enumerate(self.detected_objects):
                    x1, y1, x2, y2 = bbox
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < min_area:
                            min_area = area
                            selected_idx = idx

                if selected_idx is not None:
                    self.locked_object_index = selected_idx
                    self.lock_start_time = time.time()
                    self.has_captured = False
                    self.roi_to_save = None

                    # Nếu là ảnh tĩnh, phân loại ngay lập tức
                    if self.mode == "STATIC_IMAGE" and self.loaded_image is not None:
                        x1, y1, x2, y2 = self.detected_objects[selected_idx]
                        self.roi_to_save = self.loaded_image[y1:y2, x1:x2].copy()
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
                    if cls_id == 0:
                        continue

                    conf = float(box.conf[0])
                    if conf > 0.4:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        new_detected_objects.append((x1, y1, x2, y2))

            self.detected_objects = new_detected_objects

            if self.locked_object_index is not None and self.locked_object_index >= len(self.detected_objects):
                if len(self.detected_objects) == 0:
                    self.unlock_target()

            # Đếm ngược & phân loại khi khóa vật thể trên camera
            if self.locked_object_index is not None and self.locked_object_index < len(self.detected_objects):
                if not self.has_captured:
                    elapsed = time.time() - self.lock_start_time
                    remaining = max(0, 2 - int(elapsed))

                    if remaining > 0:
                        self.status_label.config(
                            text=f"🎯 ĐÃ KHÓA! ĐANG PHÂN TÍCH TRONG: {remaining}s...", 
                            fg="#FFEB3B"
                        )
                    else:
                        self.has_captured = True
                        x1, y1, x2, y2 = self.detected_objects[self.locked_object_index]
                        
                        y1, y2 = max(0, y1), min(self.canvas_h, y2)
                        x1, x2 = max(0, x1), min(self.canvas_w, x2)
                        
                        self.roi_to_save = frame[y1:y2, x1:x2].copy()
                        item, conf, cat, top3 = self.classify_trash(self.roi_to_save)
                        self.update_result_ui(item, conf, cat, top3)
                        self.status_label.config(text="✅ ĐÃ PHÂN LOẠI XONG!", fg="#00E676")

            # Hiển thị vẽ khung drag trên camera
            display_frame = frame.copy()
            if self.is_drawing:
                cv2.rectangle(display_frame, (self.start_x, self.start_y), (self.current_mouse_x, self.current_mouse_y), (0, 140, 255), 2)

            for idx, bbox in enumerate(self.detected_objects):
                x1, y1, x2, y2 = bbox
                if idx == self.locked_object_index:
                    color = (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, "LOCKED", (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    color = (255, 255, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)

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
