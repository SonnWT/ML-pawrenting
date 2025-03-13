# import data
from flask import Flask, render_template, Response
import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from threading import Thread, Lock
import time

# Flask initialization
app = Flask(__name__)

# Fungsi untuk mengecek apakah model tersedia
def check_model_exists(model_path):
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        return False
    return True

# Pilih device: gunakan GPU jika tersedia
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model deteksi dan emosi dengan pengecekan
model_dog = YOLO("models/dog_detection.pt").to(device) if check_model_exists("models/dog_detection.pt") else None
model_emotion_dog = YOLO("models/dog_emotion.pt").to(device) if check_model_exists("models/dog_emotion.pt") else None
model_cat = YOLO("models/cat_detection.pt").to(device) if check_model_exists("models/cat_detection.pt") else None
model_emotion_cat = YOLO("models/cat_emotion.pt").to(device) if check_model_exists("models/cat_emotion.pt") else None

# Load emoji
emoji_icons = {
    "dog": {
        "angry": cv2.imread("static/angryDog.png", cv2.IMREAD_UNCHANGED),
        "happy": cv2.imread("static/happyDog.png", cv2.IMREAD_UNCHANGED),
        "sad": cv2.imread("static/sadDog.png", cv2.IMREAD_UNCHANGED),
        "relaxed": cv2.imread("static/relaxedDog.png", cv2.IMREAD_UNCHANGED),
    },
    "cat": {
        "angry": cv2.imread("static/angryCat.png", cv2.IMREAD_UNCHANGED),
        "happy": cv2.imread("static/happyCat.png", cv2.IMREAD_UNCHANGED),
        "sad": cv2.imread("static/sadCat.png", cv2.IMREAD_UNCHANGED),
        "relaxed": cv2.imread("static/relaxedCat.png", cv2.IMREAD_UNCHANGED),
    }
}

# Threshold confidence
confidence_threshold = 0.50
emotion_confidence_threshold = 0.30

# Kelas untuk menangani video stream dalam thread terpisah
class VideoStream:
    def __init__(self, src=0):
        self.cap = None
        self.ret = None  # 🛠️ Inisialisasi ret
        self.frame = None  # 🛠️ Inisialisasi frame
        self.lock = Lock()
        self.last_access = time.time()
        self.stopped = True

    # start() → Menjalankan thread untuk membaca video dari webcam.
    def start(self):
        with self.lock:
            if self.stopped:
                self.cap = cv2.VideoCapture(0)
                self.stopped = False
                Thread(target=self.update, daemon=True).start()

    # update() → Loop untuk membaca frame video dari kamera.

    def update(self):
        while not self.stopped:
            with self.lock:
                if self.cap is not None:
                    self.ret, self.frame = self.cap.read()  # 🛠️ Pastikan membaca frame
                self.last_access = time.time()
            time.sleep(0.03)

    # read() → Mengembalikan frame terbaru.
    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()  # Gunakan `.copy()` untuk menghindari konflik thread
            return None, None  # Hindari error jika frame masih `None`

    # stop() → Menghentikan video streaming.
    def stop(self):
        with self.lock:
            self.stopped = True
            if self.cap:
                self.cap.release()
                self.cap = None

# Inisialisasi VideoStream
video_stream = VideoStream()

# Fungsi Deteksi Hewan dan Emosi
def detect_pet_and_emotion():
    # frame_skip = 2 → Lewati beberapa frame untuk mengurangi lag.
    frame_skip = 2  # Lewati 2 frame sebelum inferensi
    frame_count = 0

    # Membaca frame dari kamera.
    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("⚠️ Frame tidak bisa dibaca!")
            break

        # Melewatkan beberapa frame untuk meningkatkan kecepatan deteksi.
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frame untuk mengurangi lag

        frame_height, frame_width, _ = frame.shape
        pet_detections = []

        # Deteksi anjing
        if model_dog:
            results_dog = model_dog(frame, imgsz=320, half=True)  # Kurangi ukuran input untuk mempercepat
            for result in results_dog:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    if confidence >= confidence_threshold:
                        pet_detections.append(("dog", box))

        # Deteksi kucing
        if model_cat:
            results_cat = model_cat(frame, imgsz=320, half=True)
            for result in results_cat:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    if confidence >= confidence_threshold:
                        pet_detections.append(("cat", box))

        if not pet_detections:
            cv2.putText(frame, "Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Proses setiap hewan yang terdeteksi
        for pet_type, box in pet_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Potong area wajah hewan
            pet_face = frame[y1:y2, x1:x2]

            # Gunakan model emosi yang sesuai
            emotion_results = None
            if pet_type == "dog" and model_emotion_dog:
                emotion_results = model_emotion_dog(pet_face, imgsz=128, half=True)
            elif pet_type == "cat" and model_emotion_cat:
                emotion_results = model_emotion_cat(pet_face, imgsz=128, half=True)

            # Ambil emosi dengan confidence tertinggi
            if emotion_results and emotion_results[0].boxes:
                emotion_label = emotion_results[0].names[int(emotion_results[0].boxes.cls[0])]
                emotion_confidence = emotion_results[0].boxes.conf[0].item()

                if emotion_confidence >= emotion_confidence_threshold:
                    # Tambahkan label emosi
                    text_y = max(y1 - 10, 20)
                    cv2.putText(frame, f"{emotion_label.upper()}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                    # Tambahkan bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Tambahkan emoji
                    if emotion_label in emoji_icons[pet_type]:
                        emoji = emoji_icons[pet_type][emotion_label]
                        if emoji is not None and emoji.shape[-1] == 4:
                            emoji = cv2.resize(emoji, (70, 70))
                            emoji_x1 = x1 + (x2 - x1) // 2 - 25
                            emoji_y1 = max(y1 - 80, 0)
                            emoji_x2 = emoji_x1 + 70
                            emoji_y2 = emoji_y1 + 70

                            if 0 <= emoji_x1 < frame_width and 0 <= emoji_y1 < frame_height:
                                alpha_channel = emoji[:, :, 3] / 255.0
                                emoji_rgb = emoji[:, :, :3]
                                roi = frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2].copy()

                                for c in range(3):
                                    roi[:, :, c] = (1 - alpha_channel) * roi[:, :, c] + alpha_channel * emoji_rgb[:, :, c]
                                frame[emoji_y1:emoji_y2, emoji_x1:emoji_x2] = roi

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Menampilkan halaman utama (index.html).
@app.route('/')
def index():
    return render_template('index.html')

# Mengirim stream video ke web browser.
@app.route('/video_feed')
def video_feed():
    video_stream.start()
    return Response(detect_pet_and_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Menjalankan server Flask dan menangani Ctrl+C untuk menghentikan kamera.
if __name__ == '__main__':
    try:
        app.run(debug=False, threaded=True)
    except KeyboardInterrupt:
        video_stream.stop()
