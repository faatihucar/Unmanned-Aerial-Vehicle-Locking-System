import os
import sys

if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import math
from collections import deque
import argparse

# --- YARDIMCI FONKSİYONLAR ---
def normalized_vector(vx, vy, window_width_half, window_height_half):
    norm_x = vx / window_width_half if window_width_half != 0 else 0
    norm_y = vy / window_height_half if window_height_half != 0 else 0
    return norm_x, norm_y

def calculate_vector_magnitude(vx, vy):
    return math.sqrt(vx**2 + vy**2)

# --- GELİŞMİŞ PID KONTROLCÜSÜ SINIFI ---
class PIDController:
    def __init__(self, min_output=1000, max_output=2000):
        self.Kp, self.Ki, self.Kd, self.Kf = 0.0, 0.0, 0.0, 0.0
        self._setpoint, self._integral, self._last_error = 0.0, 0.0, 0.0
        self._last_time = time.time()
        self.min_output, self.max_output = min_output, max_output
        self._last_output = (self.min_output + self.max_output) / 2

    def update(self, measurement, dt=None):
        current_time = time.time()
        if dt is None: dt = current_time - self._last_time
        if dt == 0: return self._last_output
        error = self._setpoint - measurement
        p_term = self.Kp * error
        self._integral += self.Ki * error * dt
        i_term = self._integral
        derivative = (error - self._last_error) / dt
        d_term = self.Kd * derivative
        ff_term = self.Kf
        output = p_term + i_term + d_term + ff_term
        clamped_output = max(min(output, self.max_output), self.min_output)
        if (output != clamped_output) and (np.sign(output) == np.sign(error)):
            self._integral -= self.Ki * error * dt
        self._last_error, self._last_time, self._last_output = error, current_time, clamped_output
        return int(clamped_output)

    def set_setpoint(self, value): self._setpoint = value
    def set_gains(self, Kp, Ki, Kd, Kf=0.0):
        if self.Kp != Kp or self.Ki != Ki or self.Kd != Kd or self.Kf != Kf:
            print(f"PID Kazançları güncellendi: Kp={Kp}, Ki={Ki}, Kd={Kd}, Kf={Kf}")
            self.Kp, self.Ki, self.Kd, self.Kf = Kp, Ki, Kd, Kf
    def reset(self):
        self._integral, self.last_error = 0.0, 0.0
        self._last_time = time.time()
        self._last_output = (self.min_output + self.max_output) / 2

# --- ENTEGRE EDİLMİŞ UAVTRACKER SINIFI (REVİZE EDİLDİ) ---
class UAVTracker:
    def __init__(self, video_path, model_path="models/best.pt", output_path=None, show_display=True):
        self.device = self._select_best_device()
        print(f"🖥️ Seçilen Cihaz: {self.device.upper()}")

        if self.device == "cuda":
            print(f"🚀 GPU Modu Aktif - CUDA {torch.version.cuda}")
            print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"🖥️ CPU Modu Aktif")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened(): raise IOError(f"Video dosyasi acilamadi: {self.video_path}")

        self.frame_width, self.frame_height = 1280, 720
        self.frame_center_x, self.frame_center_y = self.frame_width // 2, self.frame_height // 2
        self.show_display = show_display

        # Video Writer Ayarları
        self.output_path = output_path
        self.writer = None
        if self.output_path:
            output_dir = os.path.dirname(os.path.abspath(self.output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or math.isnan(fps):
                fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (self.frame_width, self.frame_height))
            print(f"💾 Çıktı videosu kaydedilecek: {self.output_path} (FPS: {fps:.1f})")

        # YOLOv8s modeli yükle ve optimize et
        self.model = self._load_model_safely(model_path)

        # Model sınıflarını kontrol et ve yazdır
        print("Model siniflari:", self.model.names)

        # UAV sınıf adlarını genişlet (farklı modellerde farklı isimler olabilir)
        possible_uav_names = ["iha", "uav", "drone", "aircraft", "airplane", "plane", "fighter", "jet"]
        self.UAV_CLASS_NAME = None

        # Model sınıfları içinde UAV türü arayalım
        for class_id, class_name in self.model.names.items():
            if any(uav_name in class_name.lower() for uav_name in possible_uav_names):
                self.UAV_CLASS_NAME = class_name.lower()
                print(f"UAV sinifi bulundu: '{class_name}' (ID: {class_id})")
                break

        if self.UAV_CLASS_NAME is None:
            print("UYARI: UAV sinifi bulunamadi! Mevcut siniflar:", list(self.model.names.values()))
            print("İlk sinifi UAV olarak kabul ediyorum...")
            self.UAV_CLASS_NAME = list(self.model.names.values())[0].lower()

        # Detection parametreleri (Akademik araştırma bulgularına göre optimize)
        self.CONF_THRESHOLD = 0.2   # Ana tespit için optimal threshold (%20)
        self.TRACK_CONF_THRESHOLD = 0.15  # Tracking doğrulama için daha düşük threshold (%15)
        self.IOU_THRESHOLD = 0.4    # NMS threshold (daha iyi NMS performansı)
        self.MAX_DET = 100          # Maximum detections

        print(f"📊 Optimized Thresholds:")
        print(f"   • Ana Tespit: {self.CONF_THRESHOLD*100:.0f}%")
        print(f"   • Tracking: {self.TRACK_CONF_THRESHOLD*100:.0f}%")
        print(f"   • IOU (NMS): {self.IOU_THRESHOLD}")
        print(f"   • Max Detections: {self.MAX_DET}")

        # ROI sistemini iyileştir - daha büyük ve az ROI
        roi_width, roi_height = self.frame_width // 2, self.frame_height // 2
        self.roi_positions = [
            (0, 0),                                    # Sol üst
            (roi_width, 0),                            # Sağ üst
            (0, roi_height),                           # Sol alt
            (roi_width, roi_height)                    # Sağ alt
        ]
        self.roi_size = (roi_width, roi_height)
        self.current_roi_index = 0
        self.roi_scan_delay = 0  # ROI değişim gecikmesi
        self.ROI_SCAN_FRAMES = 12  # Optimized: Her ROI'de 12 frame bekle (daha hızlı tarama)

        print(f"🔍 ROI Tarama Ayarları:")
        print(f"   • ROI Sayısı: {len(self.roi_positions)}")
        print(f"   • ROI Boyutu: {roi_width}x{roi_height}")
        print(f"   • Frame/ROI: {self.ROI_SCAN_FRAMES}")

        self.tracking = False
        self.tracker, self.kf, self.uav_bbox = None, None, None
        self.bbox_history = deque(maxlen=3)  # Daha az history
        self.track_frame_count, self.big_enough_start_time = 0, None

        # --- DEĞİŞİKLİK BURADA ---
        self.REQUIRED_TRACKING_DURATION = 4  # Kilitlenme süresi 4 saniyeye çıkarıldı

        self.REQUIRED_BBOX_WIDTH = int(self.frame_width * 0.025)   # Optimized: %2.5 (akademik öneriler)
        self.REQUIRED_BBOX_HEIGHT = int(self.frame_height * 0.025) # Daha küçük UAV'lar için optimize

        print(f"📏 Minimum Bbox Boyutları:")
        print(f"   • Min Genişlik: {self.REQUIRED_BBOX_WIDTH}px ({self.REQUIRED_BBOX_WIDTH/self.frame_width*100:.1f}%)")
        print(f"   • Min Yükseklik: {self.REQUIRED_BBOX_HEIGHT}px ({self.REQUIRED_BBOX_HEIGHT/self.frame_height*100:.1f}%)")

        self.padding_x, self.padding_y = int(self.frame_width * 0.20), int(self.frame_height * 0.15)
        self.vurus_sayisi = 0

        self.start_time, self.frame_count = time.time(), 0

        self.altitude_padding, self.altitude_circle_radius = 60, 40
        self.altitude_circle_center = (self.frame_width - self.altitude_padding - self.altitude_circle_radius,
                                     self.frame_height - self.altitude_padding - self.altitude_circle_radius)

        target_area_percentage = 0.015  # Optimized: %1.5 (akademik öneriler için daha hassas)
        self.TARGET_BBOX_AREA = (self.frame_width * self.frame_height) * target_area_percentage
        print(f"🎯 Hedef Alan Boyutu: {self.TARGET_BBOX_AREA:.0f} piksel ({target_area_percentage*100:.1f}%)")

        self.throttle_pid = PIDController(min_output=1000, max_output=2000)
        self.gains_far = {'Kp': 300, 'Ki': 35, 'Kd': 80, 'Kf': 1480}
        self.gains_close = {'Kp': 200, 'Ki': 50, 'Kd': 120, 'Kf': 1480}
        self.throttle_pid.set_setpoint(1.0)
        self.throttle_pid.set_gains(**self.gains_far)

    def _select_best_device(self):
        """
        GPU'yu önce deneyin, sorun varsa CPU'ya geçin.
        """
        print("🔍 Cihaz seçimi yapılıyor...")

        # Önce CUDA varlığını kontrol et
        if not torch.cuda.is_available():
            print("❌ CUDA mevcut değil. CPU kullanılacak.")
            return "cpu"

        try:
            # GPU'yu test et
            print("🚀 GPU test ediliyor...")
            torch.cuda.empty_cache()  # GPU belleğini temizle

            # Basit bir GPU testi yap
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()

            print("✅ GPU test başarılı! GPU kullanılacak.")
            return "cuda"

        except Exception as e:
            print(f"❌ GPU testi başarısız: {e}")
            print("🔄 CPU'ya geçiliyor...")
            return "cpu"

    def _load_model_safely(self, model_path):
        """
        Modeli güvenli şekilde yükler ve device'a atar.
        """
        print(f"📡 Model yükleniyor: {model_path}")

        for attempt in range(3):  # 3 deneme
            try:
                print(f"🔄 Deneme {attempt + 1}/3...")

                # Model yükle
                model = YOLO(model_path)

                # Model'i device'a ata
                model.to(self.device)

                # Küçük bir test yap
                test_results = model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                                           verbose=False, device=self.device)

                print(f"✅ Model başarıyla yüklendi ve test edildi ({self.device.upper()})!")
                return model

            except Exception as e:
                print(f"❌ Model yükleme hatası (Deneme {attempt + 1}/3): {e}")

                # GPU hatası ise CPU'ya geç
                if self.device == "cuda" and attempt < 2:
                    print("🔄 GPU hatası nedeniyle CPU'ya geçiliyor...")
                    self.device = "cpu"
                    torch.cuda.empty_cache()

                if attempt < 2:
                    print("⏳ 2 saniye bekleniyor...")
                    time.sleep(2)

        raise Exception(f"❌ Model {model_path} yüklenemedi! 3 deneme başarısız.")

    def _create_kalman_filter(self, bbox):
        kf = cv2.KalmanFilter(8, 4)
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i+4] = 1.0
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 5e-3  # Daha düşük noise
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        kf.statePre = kf.statePost = state.reshape(-1, 1)
        return kf

    def _initialize_tracker(self, frame, bbox):
        # Tracker olmadan tracking (sadece YOLO + Kalman Filter)
        self.kf = self._create_kalman_filter(bbox)
        self.bbox_history.clear()
        self.bbox_history.append(bbox)
        self.tracking = True
        self.track_frame_count = 0
        self.uav_bbox = bbox
        self.big_enough_start_time = None
        self.throttle_pid.reset()
        self.last_detection_time = time.time()
        print(f"UAV tespit edildi. BBox: {bbox} - YOLO Tracking Modu")

    def _reset_tracking(self, reason=""):
        if self.tracking:
            print(f"Takip sonlandiriliyor. Neden: {reason}")
        self.tracking = False
        self.kf, self.uav_bbox = None, None
        self.current_roi_index = 0
        self.big_enough_start_time = None
        self.roi_scan_delay = 0

    def _scan_mode(self, frame):
        # ROI değişim gecikmesi
        if self.roi_scan_delay < self.ROI_SCAN_FRAMES:
            self.roi_scan_delay += 1
        else:
            self.current_roi_index = (self.current_roi_index + 1) % len(self.roi_positions)
            self.roi_scan_delay = 0

        x, y = self.roi_positions[self.current_roi_index]
        roi_w, roi_h = self.roi_size

        # ROI sınırlarını kontrol et
        roi_x_end = min(x + roi_w, self.frame_width)
        roi_y_end = min(y + roi_h, self.frame_height)
        roi = frame[y:roi_y_end, x:roi_x_end]

        if roi.size > 0:
            # YOLOv8s için optimize edilmiş inference
            results = self.model(
                roi,
                conf=self.CONF_THRESHOLD,
                iou=self.IOU_THRESHOLD,
                max_det=self.MAX_DET,
                verbose=False,
                device=self.device
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id].lower()

                    print(f"Tespit: {class_name} (conf: {confidence:.2f})")

                    # UAV sınıfını kontrol et (partial match)
                    if (confidence > self.CONF_THRESHOLD and
                        (class_name == self.UAV_CLASS_NAME or
                          any(uav_name in class_name for uav_name in ["iha", "uav", "drone", "aircraft", "airplane", "plane"]))):

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_abs = (x + x1, y + y1, x2 - x1, y2 - y1)

                        # Minimum boyut kontrolü
                        if bbox_abs[2] > 20 and bbox_abs[3] > 20:
                            self._initialize_tracker(frame, bbox_abs)
                            return

        # Mevcut ROI'yi çiz
        cv2.rectangle(frame, (x, y), (roi_x_end, roi_y_end), (0, 255, 255), 2)
        # cv2.putText(frame, f"ROI {self.current_roi_index + 1}", (x + 10, y + 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _tracking_mode(self, frame):
        # YOLO-only tracking (her frame'de detection)
        detected = False

        # Predicted region etrafında daha geniş alan ara
        pred_x, pred_y, pred_w, pred_h = self.uav_bbox
        search_margin = 100  # Arama marjı

        search_x = max(0, pred_x - search_margin)
        search_y = max(0, pred_y - search_margin)
        search_w = min(pred_w + 2*search_margin, self.frame_width - search_x)
        search_h = min(pred_h + 2*search_margin, self.frame_height - search_y)

        search_roi = frame[search_y:search_y+search_h, search_x:search_x+search_w]

        if search_roi.size > 0:
            # Arama bölgesinde YOLO çalıştır (tracking için daha düşük threshold)
            results = self.model(
                search_roi,
                conf=self.TRACK_CONF_THRESHOLD,  # Tracking için daha hassas threshold (%15)
                iou=self.IOU_THRESHOLD,
                max_det=self.MAX_DET,
                verbose=False,
                device=self.device
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                best_box = None
                best_distance = float('inf')

                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id].lower()

                    # UAV sınıfını kontrol et (tracking için daha düşük threshold)
                    if (confidence > self.TRACK_CONF_THRESHOLD and
                        (class_name == self.UAV_CLASS_NAME or
                          any(uav_name in class_name for uav_name in ["iha", "uav", "drone", "aircraft", "airplane", "plane"]))):

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_roi = (x1, y1, x2 - x1, y2 - y1)
                        bbox_abs = (search_x + x1, search_y + y1, x2 - x1, y2 - y1)

                        # En yakın detection'ı seç
                        center_x = bbox_abs[0] + bbox_abs[2] // 2
                        center_y = bbox_abs[1] + bbox_abs[3] // 2
                        pred_center_x = pred_x + pred_w // 2
                        pred_center_y = pred_y + pred_h // 2

                        distance = math.sqrt((center_x - pred_center_x)**2 + (center_y - pred_center_y)**2)
                        if distance < best_distance:
                            best_distance = distance
                            best_box = bbox_abs
                            detected = True

                if detected and best_box:
                    # Kalman filter güncelle
                    self.kf.predict()
                    self.bbox_history.append(best_box)
                    self.last_detection_time = time.time()

                    # Smooth tracking
                    if len(self.bbox_history) > 1:
                        avg_bbox = tuple(np.mean(self.bbox_history, axis=0).astype(int))
                    else:
                        avg_bbox = best_box

                    measurement = np.array([
                        avg_bbox[0] + avg_bbox[2]/2,
                        avg_bbox[1] + avg_bbox[3]/2,
                        avg_bbox[2],
                        avg_bbox[3]
                    ], dtype=np.float32).reshape(-1, 1)

                    self.kf.correct(measurement)
                    pred_cx, pred_cy, pred_w, pred_h = self.kf.statePost[:4, 0]
                    self.uav_bbox = (int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2),
                                    int(pred_w), int(pred_h))

        # Detection timeout kontrolü
        if not detected:
            if time.time() - self.last_detection_time > 2.0:  # 2 saniye timeout
                self._reset_tracking("Detection timeout")
                return
            else:
                # Sadece prediction kullan
                self.kf.predict()
                pred_cx, pred_cy, pred_w, pred_h = self.kf.statePost[:4, 0]
                self.uav_bbox = (int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2),
                                int(pred_w), int(pred_h))

        self.track_frame_count += 1

        # Alan bazlı PID kontrolü
        current_area = self.uav_bbox[2] * self.uav_bbox[3]
        if current_area <= 0:
            self._reset_tracking("Tespit edilen alan sıfır.")
            return

        normalized_area = current_area / self.TARGET_BBOX_AREA
        error_ratio = abs(1.0 - normalized_area)

        if error_ratio > 0.6:
            self.throttle_pid.set_gains(**self.gains_far)
        else:
            self.throttle_pid.set_gains(**self.gains_close)

        throttle_pwm = self.throttle_pid.update(normalized_area)

        if not self._check_tracking_validity(frame):
            return

        self._draw_tracking_elements(frame, throttle_pwm, normalized_area)

    def _check_tracking_validity(self, frame):
        pred_x, pred_y, pred_w, pred_h = self.uav_bbox

        # Sınır kontrolü
        if (pred_x < 0 or pred_y < 0 or
            pred_x + pred_w > self.frame_width or
            pred_y + pred_h > self.frame_height):
            self._reset_tracking("UAV frame sınırlarını aştı.")
            return False

        uav_within_target = (pred_x >= self.padding_x and pred_y >= self.padding_y and
                             (pred_x + pred_w) <= (self.frame_width - self.padding_x) and
                             (pred_y + pred_h) <= (self.frame_height - self.padding_y))

        if uav_within_target:
            is_big_enough = pred_w >= self.REQUIRED_BBOX_WIDTH and pred_h >= self.REQUIRED_BBOX_HEIGHT
            if is_big_enough:
                if self.big_enough_start_time is None:
                    self.big_enough_start_time = time.time()
                if time.time() - self.big_enough_start_time >= self.REQUIRED_TRACKING_DURATION:
                    self.vurus_sayisi += 1
                    print(f"BASARILI VURUS! Toplam: {self.vurus_sayisi}")
                    self._reset_tracking("Basarili vurus.")
                    return False
            else:
                self.big_enough_start_time = None
        else:
            self.big_enough_start_time = None

        # YOLO tracking modunda sürekli verification var, ek kontrol gerekmiyor
        return True

    def _draw_tracking_elements(self, frame, throttle_pwm, normalized_area):
        pred_x, pred_y, pred_w, pred_h = self.uav_bbox
        uav_within_target = (pred_x >= self.padding_x and pred_y >= self.padding_y and
                             (pred_x + pred_w) <= (self.frame_width - self.padding_x) and
                             (pred_y + pred_h) <= (self.frame_height - self.padding_y))

        # UAV bounding box'ı çiz
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 0, 255), 3)

        # Hedef alan çiz (sarı renk)
        target_zone_color = (0, 255, 255)  # BGR formatında Sarı
        cv2.rectangle(frame, (self.padding_x, self.padding_y),
                     (self.frame_width - self.padding_x, self.frame_height - self.padding_y),
                     target_zone_color, 2)

        if uav_within_target:
            status_text = ""
            text_color = (0, 255, 255)
            if self.big_enough_start_time is not None:
                duration = time.time() - self.big_enough_start_time
                status_text = f"Kilitlenme: {duration:.1f}s"
                text_color = (0, 255, 0)
            else:
                status_text = "Hedefte - Boyut Küçük"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(status_text, font, font_scale, thickness)

            status_text_pos_x = self.frame_center_x - (text_w // 2)
            status_text_pos_y = self.padding_y - 15

            cv2.putText(frame, status_text, (status_text_pos_x, status_text_pos_y),
                       font, font_scale, text_color, thickness)

        # Merkez çizgisi
        tracked_center_x, tracked_center_y = pred_x + pred_w // 2, pred_y + pred_h // 2
        cv2.line(frame, (self.frame_center_x, self.frame_center_y),
                (tracked_center_x, tracked_center_y), (0, 0, 255), 2)

        # Merkez noktaları
        cv2.circle(frame, (self.frame_center_x, self.frame_center_y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (tracked_center_x, tracked_center_y), 5, (0, 0, 255), -1)

        self._draw_control_indicators(frame, tracked_center_x, tracked_center_y, throttle_pwm, normalized_area)

    def _draw_control_indicators(self, frame, target_x, target_y, throttle_pwm, normalized_area):
        vector_x, vector_y = self.frame_center_x - target_x, self.frame_center_y - target_y
        normalized_x, normalized_y = normalized_vector(vector_x, vector_y, self.frame_width // 2, self.frame_height // 2)
        aileron_pwm = max(1000, min(int(1500 - normalized_x * 500), 2000))
        elevator_pwm = max(1000, min(int(1500 - normalized_y * 500), 2000))

        # Irtifa kontrol göstergesi
        magnitude = calculate_vector_magnitude(normalized_x, normalized_y)
        line_len = min(magnitude * self.altitude_circle_radius, self.altitude_circle_radius)
        angle = math.atan2(normalized_y, normalized_x)
        end_point = (int(self.altitude_circle_center[0] + line_len * math.cos(angle)),
                    int(self.altitude_circle_center[1] + line_len * math.sin(angle)))

        cv2.line(frame, self.altitude_circle_center, end_point, (0, 0, 255), 3)
        cv2.circle(frame, self.altitude_circle_center, self.altitude_circle_radius, (255, 255, 255), 2)
        cv2.putText(frame, "Kontrol",
                   (self.altitude_circle_center[0] - 40, self.altitude_circle_center[1] + 70),
                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 2)

        # Gaz kontrol barı
        bar_x, bar_y, bar_w, bar_h = 30, self.frame_height - 150, 200, 25
        current_ratio = min(normalized_area, 2.0) / 2.0
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * current_ratio), bar_y + bar_h), (0,0,255), -1)
        target_line_x = bar_x + int(bar_w * 0.5)
        cv2.line(frame, (target_line_x, bar_y), (target_line_x, bar_y+bar_h), (0,255,0), 3)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255,255,255), 2)
        cv2.putText(frame, f"Alan Orani: {normalized_area:.1f}", (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Kontrol değerleri
        cv2.putText(frame, f"Aileron: {aileron_pwm}", (30, self.frame_height - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Elevator: {elevator_pwm}", (30, self.frame_height - 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Throttle: {throttle_pwm}", (30, self.frame_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def _draw_overlay(self, frame):
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        # FPS ve durum bilgileri
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Vurus: {self.vurus_sayisi}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Tracking modu
        if self.tracking:
            mode_text = "YOLO TAKIP MODU"
            mode_color = (0, 255, 0)
        else:
            mode_text = f"TARAMA MODU (ROI {self.current_roi_index + 1}/4)"
            mode_color = (0, 255, 255)
        cv2.putText(frame, mode_text, (10, self.frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    def run(self):
        print("UAV Tracker başlatılıyor...")
        if self.show_display:
            print("Çıkmak için 'q' tuşuna basın")

        with torch.no_grad():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Video sonuna ulaşıldı veya okuma tamamlandı.")
                    break

                # Frame'i resize et
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                self.frame_count += 1

                # Ana işlem
                if self.tracking:
                    self._tracking_mode(frame)
                else:
                    self._scan_mode(frame)

                # Overlay çiz
                self._draw_overlay(frame)

                # Video dosyasına yaz
                if self.writer is not None:
                    self.writer.write(frame)

                # Frame'i ekranda göster
                if self.show_display:
                    cv2.imshow("YOLOv8s UAV Tracker", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Kullanıcı tarafından durduruldu.")
                        break
                    elif key == ord('r'):  # Reset tracking
                        self._reset_tracking("Manuel reset")
                    elif key == ord('s'):  # Screenshot
                        cv2.imwrite(f"screenshot_{int(time.time())}.jpg", frame)
                        print("Ekran görüntüsü kaydedildi")

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
            print(f"🎬 İşlenen video başarıyla kaydedildi: {self.output_path}")
        if self.show_display:
            cv2.destroyAllWindows()
        print("Kaynaklar serbest bırakıldı.")

# --- ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="YOLOv8s UAV Tracker")
    parser.add_argument("--video", type=str, default=os.path.join(project_root, "videos", "sahne1.mp4"), help="Video dosya yolu")
    parser.add_argument("--model", type=str, default=os.path.join(project_root, "models", "best.pt"), help="YOLO model dosya yolu")
    parser.add_argument("--output", type=str, default=os.path.join(project_root, "result", "sahne1_result.mp4"), help="Sonuç video kayıt yolu")
    parser.add_argument("--no-show", action="store_true", help="Görsel pencereyi açmadan (headless) çalıştır")

    args = parser.parse_args()

    print("=== YOLOv8s UAV Tracker ===")
    print(f"Video : {args.video}")
    print(f"Model : {args.model}")
    print(f"Çıktı : {args.output}")

    try:
        uav_tracker_app = UAVTracker(
            video_path=args.video,
            model_path=args.model,
            output_path=args.output,
            show_display=not args.no_show
        )
        uav_tracker_app.run()
    except FileNotFoundError as e:
        print(f"Dosya bulunamadı: {e}")
    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()