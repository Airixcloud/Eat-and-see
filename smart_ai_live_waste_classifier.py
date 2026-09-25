import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import json
import base64
import urllib.request
import urllib.error
import os
import csv
from datetime import datetime

# Optional Text-To-Speech library fallback support
try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

SYSTEM_PROMPT = """
Bạn là chuyên gia AI phân loại rác thải môi trường thời gian thực từ camera live.
Hãy phân tích hình ảnh và trả về DUY NHẤT một chuỗi JSON hợp lệ (không kèm theo bất kỳ văn bản nào khác hay thẻ markdown ```json) với cấu trúc:
{
  "title": "Tên rác thải/vật thể nhận diện được (Tiếng Việt)",
  "trashType": "Hữu cơ" | "Tái chế" | "Vô cơ" | "Nguy hại",
  "confidence": 95,
  "binColor": "Xanh Lá" | "Xanh Dương" | "Xám" | "Đỏ",
  "binName": "Thùng Rác Hữu Cơ" | "Thùng Rác Tái Chế" | "Thùng Rác Vô Cơ/Khác" | "Thùng Rác Nguy Hại",
  "description": "Mô tả ngắn gọn đặc tính vật liệu và lý do phân loại.",
  "instructions": "Hướng dẫn xử lý ngắn (Ví dụ: Rửa sạch chai nhựa trước khi vứt)"
}
"""

TRASH_THEMES = {
    "Hữu cơ": {"bg": "#15803d", "fg": "#86efac", "icon": "🌿", "bin": "Thùng Xanh Lá"},
    "Tái chế": {"bg": "#1d4ed8", "fg": "#93c5fd", "icon": "♻️", "bin": "Thùng Xanh Dương"},
    "Vô cơ": {"bg": "#334155", "fg": "#cbd5e1", "icon": "🗑️", "bin": "Thùng Xám / Đen"},
    "Nguy hại": {"bg": "#b91c1c", "fg": "#fca5a5", "icon": "⚠️", "bin": "Thùng Đỏ / Vàng"}
}

class VoiceAssistant:
    """Manages offline audio announcements for waste classification."""
    def __init__(self):
        self.engine = None
        self.enabled = HAS_TTS
        if HAS_TTS:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 160)
            except Exception as e:
                print(f"[TTS Warning] Could not initialize pyttsx3: {e}")
                self.enabled = False

    def speak(self, text):
        if not self.enabled or not self.engine:
            return
        
        def _speak_thread():
            try:
                # Re-initialize inside thread if needed for stability
                engine = pyttsx3.init()
                engine.setProperty('rate', 165)
                engine.say(text)
                engine.runAndWait()
            except Exception as err:
                print(f"[TTS Error] Speech playback failed: {err}")

        threading.Thread(target=_speak_thread, daemon=True).start()

class GeminiAIWorker:
    """Handles asynchronous image classification requests via Gemini API."""
    def __init__(self, api_key=""):
        self.api_key = api_key

    def classify_frame(self, frame_bgr, callback):
        """Sends an OpenCV frame to Gemini API in a background thread."""
        def _process():
            if not self.api_key:
                # Return simulated fallback response if API key is not set
                time.sleep(0.8)
                result = {
                    "title": "Chai Nhựa PET (Chế độ Demo)",
                    "trashType": "Tái chế",
                    "confidence": 92,
                    "binColor": "Xanh Dương",
                    "binName": "Thùng Rác Tái Chế",
                    "description": "Chai nhựa đồ uống trong suốt. Nhập Gemini API Key ở bảng cài đặt để bật AI thật.",
                    "instructions": "Bóp dẹp chai nhựa và vặn chặt nắp trước khi bỏ vào thùng rác tái chế."
                }
                callback(True, result)
                return

            try:
                # Convert BGR frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                base64_image = base64.b64encode(buffer).decode('utf-8')

                # Using recommended Gemini 2.5 Flash endpoint
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": SYSTEM_PROMPT},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }]
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )

                with urllib.request.urlopen(req, timeout=10) as response:
                    res_body = json.loads(response.read().decode('utf-8'))
                    candidate_text = res_body['candidates'][0]['content']['parts'][0]['text'].trim()

                    # Clean potential markdown wrapping
                    if candidate_text.startswith("```json"):
                        candidate_text = candidate_text[7:]
                    if candidate_text.startswith("```"):
                        candidate_text = candidate_text[3:]
                    if candidate_text.endswith("```"):
                        candidate_text = candidate_text[:-3]

                    parsed_json = json.loads(candidate_text.strip())
                    callback(True, parsed_json)

            except Exception as err:
                print(f"[AI Error] Classification failed: {err}")
                callback(False, str(err))

        threading.Thread(target=_process, daemon=True).start()

class SmartWasteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart AI Live Waste Classifier - Quét & Phân Loại Rác Trực Tiếp")
        self.root.geometry("1280x820")
        self.root.configure(bg="#0f172a")

        # Application State Variables
        self.api_key_var = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))
        self.auto_scan_var = tk.BooleanVar(value=True)
        self.speech_enabled_var = tk.BooleanVar(value=True)
        self.scan_interval_var = tk.DoubleVar(value=2.5) # Seconds between AI scans
        
        self.camera_idx = 0
        self.cap = None
        self.is_running = True
        self.is_analyzing = False
        self.last_scan_time = 0
        
        self.ai_worker = GeminiAIWorker(self.api_key_var.get())
        self.voice = VoiceAssistant()

        # Analytics Counter
        self.stats = {"Hữu cơ": 0, "Tái chế": 0, "Vô cơ": 0, "Nguy hại": 0, "Tổng cộng": 0}
        self.history_logs = []

        # Build User Interface Components
        self._setup_styles()
        self._build_ui()
        self._start_camera()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure dark themes
        self.style.configure(".", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 10))
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("Card.TFrame", background="#1e293b", relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#10b981", background="#0f172a")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground="#94a3b8", background="#1e293b")
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#10b981", foreground="#ffffff")
        self.style.map("Primary.TButton", background=[("active", "#059669")])

    def _build_ui(self):
        # Top Header Bar
        header_frame = ttk.Frame(self.root, padding=(15, 10))
        header_frame.pack(fill="x")
        
        ttk.Label(header_frame, text="♻️ SMART AI WASTE CLASSIFIER (LIVE CAMERA)", style="Header.TLabel").pack(side="left")
        
        self.status_pill = tk.Label(
            header_frame, text="● LIVE SCANNING ACTIVE", font=("Segoe UI", 9, "bold"),
            bg="#064e3b", fg="#34d399", padx=10, pady=4
        )
        self.status_pill.pack(side="right")

        # Main Content Layout (Split Left - Right)
        content = ttk.Frame(self.root, padding=10)
        content.pack(fill="both", expand=True)

        # Left Column: Video Live Stream Canvas
        left_col = ttk.Frame(content)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(left_col, bg="#020617", highlightthickness=1, highlightbackground="#334155")
        self.canvas.pack(fill="both", expand=True)

        # Quick Control Overlay under video
        vid_controls = ttk.Frame(left_col, padding=(0, 10))
        vid_controls.pack(fill="x")

        self.btn_auto = tk.Button(
            vid_controls, text="🔄 Đang Tự Động Quét", font=("Segoe UI", 9, "bold"),
            bg="#10b981", fg="white", activebackground="#059669",
            command=self._toggle_auto_mode, relief="flat", padx=12, pady=6
        )
        self.btn_auto.pack(side="left", padx=5)

        tk.Button(
            vid_controls, text="📷 Quét Ngay Tức Thì", font=("Segoe UI", 9, "bold"),
            bg="#3b82f6", fg="white", activebackground="#1d4ed8",
            command=self._trigger_manual_scan, relief="flat", padx=12, pady=6
        ).pack(side="left", padx=5)

        tk.Button(
            vid_controls, text="⚙️ Đổi Camera", font=("Segoe UI", 9),
            bg="#334155", fg="white", activebackground="#475569",
            command=self._switch_camera, relief="flat", padx=10, pady=6
        ).pack(side="right")

        # Right Column: Dashboard & AI Classification Results
        right_col = ttk.Frame(content, width=420)
        right_col.pack(side="right", fill="both", expand=False)
        right_col.pack_propagate(False)

        # Card 1: AI Analysis Result
        res_card = ttk.Frame(right_col, style="Card.TFrame", padding=15)
        res_card.pack(fill="x", pady=(0, 10))

        ttk.Label(res_card, text="KẾT QUẢ NHẬN DIỆN AI", style="SubHeader.TLabel").pack(anchor="w")

        # Result Title & Type Badge
        self.lbl_item_name = tk.Label(
            res_card, text="Đang đợi đưa rác vào camera...", font=("Segoe UI", 15, "bold"),
            bg="#1e293b", fg="#f8fafc", anchor="w", wraplength=380, justify="left"
        )
        self.lbl_item_name.pack(fill="x", pady=(8, 4))

        self.badge_type = tk.Label(
            res_card, text="TRẠNG THÁI: CHỜ QUÉT", font=("Segoe UI", 9, "bold"),
            bg="#334155", fg="#cbd5e1", padx=8, pady=3
        )
        self.badge_type.pack(anchor="w", pady=(0, 10))

        # Recommended Disposal Bin Card
        self.bin_box = tk.Frame(
    res_card,
    bg="#0f172a",
    highlightthickness=1,
    highlightbackground="#334155",
    padx=10,
    pady=10
)
        self.bin_box.pack(fill="x", pady=5)

        self.lbl_bin_icon = tk.Label(self.bin_box, text="🔍", font=("Segoe UI", 24), bg="#0f172a", fg="#10b981")
        self.lbl_bin_icon.pack(side="left", padx=(0, 10))

        bin_text_frame = tk.Frame(self.bin_box, bg="#0f172a")
        bin_text_frame.pack(side="left", fill="both", expand=True)

        self.lbl_bin_name = tk.Label(bin_text_frame, text="Hướng Dẫn Thùng Rác", font=("Segoe UI", 11, "bold"), bg="#0f172a", fg="#10b981", anchor="w")
        self.lbl_bin_name.pack(fill="x")

        self.lbl_instructions = tk.Label(bin_text_frame, text="Đưa vật thể rác thải vào tâm khung quét để AI tự động phân tích.", font=("Segoe UI", 8), bg="#0f172a", fg="#94a3b8", anchor="w", wraplength=280, justify="left")
        self.lbl_instructions.pack(fill="x")

        # Material Description
        self.lbl_desc = tk.Label(
            res_card, text="Mô tả chi tiết vật liệu sẽ hiển thị ở đây.", font=("Segoe UI", 9),
            bg="#1e293b", fg="#cbd5e1", wraplength=380, justify="left"
        )
        self.lbl_desc.pack(fill="x", pady=(8, 0))

        # Card 2: Analytics Stats
        stats_card = ttk.Frame(right_col, style="Card.TFrame", padding=15)
        stats_card.pack(fill="x", pady=(0, 10))

        ttk.Label(stats_card, text="THỐNG KÊ PHÂN LOẠI PHIÊN NÀY", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 8))

        stats_grid = tk.Frame(stats_card, bg="#1e293b")
        stats_grid.pack(fill="x")

        self.stat_labels = {}
        items = [("Hữu cơ", "🌿", "#22c55e"), ("Tái chế", "♻️", "#3b82f6"), ("Vô cơ", "🗑️", "#94a3b8"), ("Nguy hại", "⚠️", "#ef4444")]
        
        for idx, (name, icon, color) in enumerate(items):
            row, col = divmod(idx, 2)
            f = tk.Frame(stats_grid, bg="#0f172a", padx=6, pady=6)
            f.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
            stats_grid.columnconfigure(col, weight=1)

            tk.Label(f, text=f"{icon} {name}", font=("Segoe UI", 8, "bold"), bg="#0f172a", fg=color).pack(anchor="w")
            lbl_val = tk.Label(f, text="0", font=("Segoe UI", 12, "bold"), bg="#0f172a", fg="#f8fafc")
            lbl_val.pack(anchor="e")
            self.stat_labels[name] = lbl_val

        # Card 3: System Settings & Configurations
        cfg_card = ttk.Frame(right_col, style="Card.TFrame", padding=15)
        cfg_card.pack(fill="both", expand=True)

        ttk.Label(cfg_card, text="CẤU HÌNH HE THONG", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 8))

        # API Key Input
        tk.Label(cfg_card, text="Gemini API Key:", font=("Segoe UI", 8, "bold"), bg="#1e293b", fg="#94a3b8").pack(anchor="w")
        api_entry = tk.Entry(cfg_card, textvariable=self.api_key_var, show="•", bg="#0f172a", fg="white", insertbackground="white", relief="flat")
        api_entry.pack(fill="x", pady=(2, 8))
        api_entry.bind("<FocusOut>", lambda e: self._update_api_key())

        # Scan Interval Slider
        tk.Label(cfg_card, text="Tần suất quét live (Giây):", font=("Segoe UI", 8, "bold"), bg="#1e293b", fg="#94a3b8").pack(anchor="w")
        slider = ttk.Scale(cfg_card, from_=1.0, to=5.0, variable=self.scan_interval_var, orient="horizontal")
        slider.pack(fill="x", pady=(2, 8))

        # Audio Toggle Checkbox
        chk_speech = tk.Checkbutton(
            cfg_card, text="🔊 Bật Giọng Nói Đọc Kết Quả (TTS)", variable=self.speech_enabled_var,
            bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a", activebackground="#1e293b", activeforeground="white"
        )
        chk_speech.pack(anchor="w", pady=(0, 8))

        # Export Report Button
        tk.Button(
            cfg_card, text="📊 Xuất Báo Cáo CSV", font=("Segoe UI", 9, "bold"),
            bg="#0284c7", fg="white", activebackground="#0369a1",
            command=self._export_csv, relief="flat", pady=5
        ).pack(fill="x")

    def _start_camera(self):
        """Initializes OpenCV VideoCapture stream."""
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi Camera", f"Không thể mở camera thiết bị chỉ số #{self.camera_idx}")
            return
        
        self.root.after(10, self._update_video_feed)

    def _update_video_feed(self):
        """Continuous video loop for drawing HUD and handling auto-scans."""
        if not self.is_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # Mirror frame horizontally for natural view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Draw Viewfinder Reticle Box & HUD on OpenCV Frame
            box_size = int(min(h, w) * 0.55)
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Draw Target Reticle Corners
            color = (16, 185, 129) if not self.is_analyzing else (59, 130, 246) # Green vs Blue when analyzing
            thickness = 3
            corner_len = 25

            # Reticle Box Corners
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

            # Animated Scanning Line inside box
            if self.auto_scan_var.get() or self.is_analyzing:
                scan_line_y = y1 + int((time.time() * 200) % box_size)
                cv2.line(frame, (x1 + 5, scan_line_y), (x2 - 5, scan_line_y), (52, 211, 153), 2)

            # Check for Continuous Auto Live-Scanning Trigger
            now = time.time()
            if self.auto_scan_var.get() and not self.is_analyzing:
                if now - self.last_scan_time >= self.scan_interval_var.get():
                    self._request_ai_scan(frame)

            # Convert BGR to RGB for Tkinter Canvas display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize image dynamically to fit Tkinter Canvas
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                img = img.resize((cw, ch), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.root.after(30, self._update_video_feed)

    def _request_ai_scan(self, frame_bgr):
        """Triggers asynchronous AI analysis."""
        self.is_analyzing = True
        self.last_scan_time = time.time()
        self.status_pill.config(text="⏳ AI ĐANG PHÂN PHAT...", bg="#1e3a8a", fg="#93c5fd")
        
        self.ai_worker.classify_frame(frame_bgr, self._on_ai_result_received)

    def _on_ai_result_received(self, success, data):
        """Callback fired when background AI worker returns result."""
        self.is_analyzing = False
        self.status_pill.config(text="● LIVE SCANNING ACTIVE", bg="#064e3b", fg="#34d399")

        if not success:
            print(f"API Error: {data}")
            return

        # Update GUI UI with parsed classification data
        trash_type = data.get("trashType", "Vô cơ")
        theme = TRASH_THEMES.get(trash_type, TRASH_THEMES["Vô cơ"])

        self.lbl_item_name.config(text=data.get("title", "Vật thể không rõ"))
        self.badge_type.config(
            text=f"LOẠI: {trash_type.upper()} ({data.get('confidence', 90)}% TIN CẬY)",
            bg=theme["bg"], fg=theme["fg"]
        )

        self.lbl_bin_icon.config(text=theme["icon"])
        self.lbl_bin_name.config(text=theme["bin"], fg=theme["fg"])
        self.lbl_instructions.config(text=data.get("instructions", "Phân loại vào thùng rác phù hợp."))
        self.lbl_desc.config(text=data.get("description", ""))

        # Update Statistics Counters
        if trash_type in self.stats:
            self.stats[trash_type] += 1
            self.stats["Tổng cộng"] += 1
            self.stat_labels[trash_type].config(text=str(self.stats[trash_type]))

        # Log entry for history
        self.history_logs.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": data.get("title", ""),
            "trashType": trash_type,
            "confidence": data.get("confidence", 0),
            "bin": theme["bin"]
        })

        # Voice Announcement Trigger
        if self.speech_enabled_var.get():
            spoken_text = f"{data.get('title')}. Hãy bỏ vào {theme['bin']}."
            self.voice.speak(spoken_text)

    def _toggle_auto_mode(self):
        """Switches between continuous auto-scanning and manual trigger mode."""
        new_state = not self.auto_scan_var.get()
        self.auto_scan_var.set(new_state)

        if new_state:
            self.btn_auto.config(text="🔄 Đang Tự Động Quét", bg="#10b981")
        else:
            self.btn_auto.config(text="⏸️ Đã Tắt Tự Động", bg="#64748b")

    def _trigger_manual_scan(self):
        """Forces an immediate manual scan frame capture."""
        if self.cap and not self.is_analyzing:
            ret, frame = self.cap.read()
            if ret:
                self._request_ai_scan(frame)

    def _switch_camera(self):
        """Toggles to the next camera index."""
        if self.cap:
            self.cap.release()
        self.camera_idx = 0 if self.camera_idx == 1 else 1
        self._start_camera()

    def _update_api_key(self):
        """Updates API key in worker service."""
        key = self.api_key_var.get().strip()
        self.ai_worker.api_key = key

    def _export_csv(self):
        """Exports session classification stats and history to a CSV file."""
        if not self.history_logs:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu lịch sử nhận diện để xuất file!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Lưu Báo Cáo Phân Loại Rác"
        )
        if file_path:
            try:
                with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=["timestamp", "title", "trashType", "confidence", "bin"])
                    writer.writeheader()
                    writer.writerows(self.history_logs)
                messagebox.showinfo("Thành công", f"Đã xuất báo cáo CSV thành công tới:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file: {e}")

    def on_close(self):
        """Clean shutdown releases camera resources."""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartWasteApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()