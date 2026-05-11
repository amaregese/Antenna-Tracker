import time

import cv2
from ultralytics import YOLO

import config
from .matching import (
    clamp_box,
    extract_features,
    score_detection_match,
    score_reacquisition_match,
)


class Detection:
    def __init__(self, left, top, right, bottom, class_id, class_name, confidence):
        self.Left = left
        self.Top = top
        self.Right = right
        self.Bottom = bottom
        self.Center = ((left + right) // 2, (top + bottom) // 2)
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.is_selected = False
        self.width = max(0, right - left)
        self.height = max(0, bottom - top)
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height if self.height > 0 else 1.0


class Detector:
    def __init__(self, model_path="models/yolo11n.pt"):
        self.model = YOLO(model_path)
        self.classes = self.model.names if hasattr(self.model, 'names') else {}
        print(f"Model loaded: {len(self.classes)} classes")

        self.selected_object = None
        self.selected_class = None
        self.selected_features = {}
        self.target_memory = {}
        self.last_known_object = None
        self.last_known_center = None
        self.last_known_velocity = (0.0, 0.0)
        self.lost_frame_count = 0
        self.tracking_lost = False
        self.tracking_confidence = 0
        self.last_frame = None
        self.frame_counter = 0
        self.last_fps_time = time.perf_counter()
        self.current_fps = 0.0

    def _predict(self, frame):
        results = self.model.predict(
            source=frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            verbose=False,
        )

        detections = []
        h, w = frame.shape[:2]
        min_area = h * w * config.MIN_BOX_AREA_RATIO

        if results[0].boxes is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = clamp_box(*box[:4], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = self.classes.get(cls_id, f"class_{cls_id}")
            det = Detection(x1, y1, x2, y2, cls_id, class_name, float(conf))
            if det.area < min_area:
                continue
            detections.append(det)

        return detections

    def _pick_unique(self, scored_candidates, min_score, margin):
        if not scored_candidates:
            return None, 0.0
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_match = scored_candidates[0]
        if best_score < min_score:
            return None, best_score
        if len(scored_candidates) > 1:
            if best_score - scored_candidates[1][0] < margin:
                return None, best_score
        return best_match, best_score

    def _update_tracking(self, detections, frame):
        self.tracking_confidence = 0
        if not self.selected_class:
            self.tracking_lost = False
            return

        same_class = [d for d in detections if d.class_name == self.selected_class]

        if self.selected_object is not None and same_class:
            ref_features = self.selected_features or extract_features(
                self.last_frame if self.last_frame is not None else frame, self.selected_object
            )
            candidates = [(score_detection_match(self.selected_object, ref_features, d, frame), d) for d in same_class]
            best_match, best_score = self._pick_unique(
                candidates, config.TRACK_MATCH_THRESHOLD, config.TRACK_UNIQUENESS_MARGIN
            )
            self.tracking_confidence = min(100.0, best_score)
            if best_match is not None:
                prev_center = self.last_known_center or self.selected_object.Center
                best_match.is_selected = True
                self.selected_object = best_match
                self.selected_features = extract_features(frame, best_match)
                self.target_memory = dict(self.selected_features)
                self.last_known_object = best_match
                self.last_known_velocity = (
                    best_match.Center[0] - prev_center[0],
                    best_match.Center[1] - prev_center[1],
                )
                self.last_known_center = best_match.Center
                self.lost_frame_count = 0
                self.tracking_lost = False
                return

        if self.selected_object is not None:
            self.selected_object.is_selected = False
            self.last_known_object = self.selected_object
            self.last_known_center = self.selected_object.Center
        self.selected_object = None
        self.selected_features = {}
        self.tracking_lost = True
        self.lost_frame_count += 1

        if not same_class or self.lost_frame_count > config.TARGET_MEMORY_FRAMES:
            return

        use_predicted = self.lost_frame_count < 30
        predicted_center = None
        if use_predicted and self.last_known_center is not None:
            predicted_center = (
                int(self.last_known_center[0] + self.last_known_velocity[0] * min(self.lost_frame_count, 10)),
                int(self.last_known_center[1] + self.last_known_velocity[1] * min(self.lost_frame_count, 10)),
            )

        memory = self.target_memory or (
            extract_features(self.last_frame if self.last_frame is not None else frame, self.last_known_object)
            if self.last_known_object is not None else {}
        )
        if not memory:
            return

        candidates = []
        frame_diag = (frame.shape[1] ** 2 + frame.shape[0] ** 2) ** 0.5
        for d in same_class:
            ratio = min(memory["area"], d.area) / max(memory["area"], d.area, 1)
            if ratio < config.MIN_REACQUIRE_AREA_RATIO:
                continue
            if use_predicted and predicted_center is not None:
                dx = d.Center[0] - predicted_center[0]
                dy = d.Center[1] - predicted_center[1]
                if (dx * dx + dy * dy) ** 0.5 > frame_diag * config.MAX_REACQUIRE_CENTER_SHIFT_RATIO:
                    continue
            score = score_reacquisition_match(memory, d, frame, predicted_center)
            candidates.append((score, d))

        best_match, best_score = self._pick_unique(
            candidates, config.REACQUIRE_MATCH_THRESHOLD, config.REACQUIRE_UNIQUENESS_MARGIN
        )
        self.tracking_confidence = min(100.0, best_score)
        if best_match is None:
            return

        best_match.is_selected = True
        self.selected_object = best_match
        self.selected_features = extract_features(frame, best_match)
        self.target_memory = dict(self.selected_features)
        if self.last_known_center is not None:
            self.last_known_velocity = (
                best_match.Center[0] - self.last_known_center[0],
                best_match.Center[1] - self.last_known_center[1],
            )
        self.last_known_center = best_match.Center
        self.tracking_lost = False
        self.lost_frame_count = 0

    def detect(self, frame):
        self.frame_counter += 1
        skip = config.FRAME_SKIP
        if self.frame_counter % skip != 0 and hasattr(self, '_last_detections') and self._last_detections:
            return self._last_detections

        detections = self._predict(frame)
        self._update_tracking(detections, frame)

        self._last_detections = detections
        self.last_frame = frame.copy()
        return detections

    def process_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return [], 0.0, None

        if frame.shape[1] > config.DEFAULT_WIDTH:
            scale = config.DEFAULT_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (config.DEFAULT_WIDTH, int(frame.shape[0] * scale)))

        detections = self.detect(frame)

        now = time.perf_counter()
        elapsed = now - self.last_fps_time
        if elapsed > 0:
            self.current_fps = 1.0 / elapsed
        self.last_fps_time = now

        return detections, self.current_fps, frame

    def find_object_at(self, detections, x, y):
        hits = [d for d in detections if d.Left <= x <= d.Right and d.Top <= y <= d.Bottom]
        if not hits:
            return None
        hits.sort(key=lambda o: o.area)
        return hits[0]

    def select_object(self, obj):
        if obj is None:
            return False
        if self.selected_object:
            self.selected_object.is_selected = False
        self.selected_class = obj.class_name
        self.selected_object = obj
        obj.is_selected = True
        ref_frame = self.last_frame
        self.selected_features = extract_features(ref_frame, obj)
        self.target_memory = dict(self.selected_features)
        self.last_known_object = obj
        self.last_known_center = obj.Center
        self.last_known_velocity = (0.0, 0.0)
        self.lost_frame_count = 0
        self.tracking_lost = False
        self.tracking_confidence = obj.confidence * 100
        print(f"Selected: {self.selected_class} ({obj.confidence * 100:.1f}%)")
        return True

    def clear_selection(self):
        if self.selected_object:
            self.selected_object.is_selected = False
        self.selected_class = None
        self.selected_object = None
        self.selected_features = {}
        self.target_memory = {}
        self.last_known_object = None
        self.last_known_center = None
        self.last_known_velocity = (0.0, 0.0)
        self.lost_frame_count = 0
        self.tracking_lost = False
        self.tracking_confidence = 0

    def get_selected(self):
        return self.selected_object

    def get_tracking_status(self):
        return self.tracking_lost

    def get_tracking_confidence(self):
        return self.tracking_confidence

    def cleanup(self):
        cv2.destroyAllWindows()
