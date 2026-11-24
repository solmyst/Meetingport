"""MediaPipe wrappers (segmentation, hands, face) + clean-plate builder + legacy differencing fallback."""
import os, urllib.request, cv2, numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODELS = {
    "hand": ("hand_landmarker.task", "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
    "seg": ("selfie_segmenter.tflite", "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
    "face": ("blaze_face_short_range.tflite", "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"),
}

def model_path(key):
    name, url = MODELS[key]; p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        os.makedirs(MODEL_DIR, exist_ok=True); print(f"downloading {name} ..."); urllib.request.urlretrieve(url, p)
    return p

def _mp():
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    return mp, vision, BaseOptions

def _img(mp, frame_bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))

class MPSegmenter:
    def __init__(self, smooth=0.5):
        mp, vision, BO = _mp(); self.mp = mp; self.prev = None; self.smooth = smooth
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BO(model_asset_path=model_path("seg")), running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False))
    def __call__(self, frame, ts_ms, plate=None):
        m = self.seg.segment_for_video(_img(self.mp, frame), ts_ms).confidence_masks[0].numpy_view().astype(np.float32)
        if m.ndim == 3: m = m[..., 0]
        m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
        m = np.clip((m - 0.35) / 0.3, 0, 1)                     # sharpen soft edges
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))            # cover hair/edge halo
        m = cv2.GaussianBlur(m, (0, 0), 2)
        if self.prev is not None and self.prev.shape == m.shape: m = self.smooth * self.prev + (1 - self.smooth) * m
        self.prev = m; return m

class DiffSegmenter:
    """v1 fallback: difference against the clean plate. Needs a static camera and stable light."""
    def __init__(self):
        self.prev = None
    def __call__(self, frame, ts_ms, plate=None, thr=35):
        if plate is None:
            self.prev = np.zeros(frame.shape[:2], np.float32)
            return self.prev
        d = cv2.absdiff(cv2.GaussianBlur(frame, (7, 7), 0), cv2.GaussianBlur(plate.astype(np.uint8), (7, 7), 0)).max(axis=2)
        m = (d > thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), None, iterations=2)
        m = cv2.GaussianBlur(m, (21, 21), 0).astype(np.float32) / 255
        self.prev = m
        return m

class HandTracker:
    def __init__(self, num_hands=2, min_conf=0.6):
        mp, vision, BO = _mp(); self.mp = mp
        self.det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BO(model_asset_path=model_path("hand")), running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf, min_tracking_confidence=0.5))
    def __call__(self, frame, ts_ms):
        """Hands as lists of 21 (x, y) in units of image HEIGHT (so x and y distances are comparable)."""
        r = self.det.detect_for_video(_img(self.mp, frame), ts_ms); asp = frame.shape[1] / frame.shape[0]
        return [[(p.x * asp, p.y) for p in h] for h in r.hand_landmarks]

class FaceTracker:
    def __init__(self):
        mp, vision, BO = _mp(); self.mp = mp
        self.det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BO(model_asset_path=model_path("face")), running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.5))
    def __call__(self, frame, ts_ms):
        """(forehead_point, face_width) in image-height units, or (None, None)."""
        r = self.det.detect_for_video(_img(self.mp, frame), ts_ms); H = frame.shape[0]
        if not r.detections: return None, None
        b = max(r.detections, key=lambda d: d.bounding_box.width).bounding_box
        return ((b.origin_x + b.width / 2) / H, (b.origin_y + b.height * 0.12) / H), b.width / H

class PlateCapture:
    """Clean empty-room capture that tolerates you being in some frames: each pixel is averaged only over
    frames where the (dilated) person mask is clear there, so nobody gets baked into the plate.
    Feed frames with add(); done() once every pixel has enough clean samples (or max_frames is hit)."""
    def __init__(self, shape, target_frames, min_clean=None, max_factor=3.0, dilate_px=None):
        h, w = shape[:2]
        self.sum = np.zeros((h, w, 3), np.float32); self.n = np.zeros((h, w), np.float32)
        self.frames, self.target = 0, max(1, int(target_frames))
        self.min_clean = min_clean or max(1, min(15, self.target // 2))
        self.max_frames = int(self.target * max_factor)
        k = dilate_px or max(15, int(w * 0.045)) | 1               # ~57 px at 1280: also covers soft edges + nearby shadow
        self.kernel = np.ones((k, k), np.uint8); self.coverage = 0.0
    def add(self, frame, mask):
        self.coverage = float((mask > 0.5).mean()) if mask is not None else 0.0
        self.seen = getattr(self, "seen", 0) + 1
        if self.person_in_frame(): return False                    # whole frame rejected while you're in view
        self.frames += 1
        if mask is None: clean = np.ones(self.n.shape, np.float32)
        else: clean = 1.0 - cv2.dilate((mask > 0.05).astype(np.uint8), self.kernel).astype(np.float32)
        self.sum += frame.astype(np.float32) * clean[..., None]; self.n += clean; return True
    def clean_fraction(self): return float((self.n >= self.min_clean).mean())
    def person_in_frame(self, thr=0.02): return self.coverage > thr
    def done(self):
        if self.frames < self.target or self.person_in_frame(): return False   # never finishes while you're seen
        return self.clean_fraction() >= 0.995 or self.frames >= self.max_frames
    def result(self):
        """(plate, known): known = 1 where at least one clean sample exists; the rest gets inpainted/learned later."""
        known = (self.n >= 1).astype(np.float32)
        return self.sum / np.maximum(self.n, 1)[..., None], known

class PlateBuilder:
    """Clean plate = the room without you. Seeded by an empty-room capture, then continuously refreshed
    from pixels the segmenter says are background, so lighting drift is tracked. Holes are inpainted.
    Refresh is gated: a known pixel only drifts toward the camera when it is already close to the plate,
    so your cast shadow (or a soft mask edge) next to you is never learned into the empty room."""
    def __init__(self, shape):
        self.plate = np.zeros(shape, np.float32); self.known = np.zeros(shape[:2], np.float32)
    def seed(self, frames):
        self.plate = np.median(np.stack(frames), axis=0).astype(np.float32); self.known[:] = 1
    def seed_capture(self, cap):
        self.plate, self.known = cap.result()
        if self.known.min() == 0:                                   # never-clean pixels: inpaint now, refine later
            self.plate = self.get()
    def update(self, frame, mask, rate=0.05, gate=28.0):
        f = frame.astype(np.float32)
        bg = mask < 0.05
        close = np.abs(f - self.plate).max(axis=2) < gate          # lighting drift only, not shadows / people
        learn = bg & ((self.known == 0) | close)
        a = (learn * np.where(self.known > 0, rate, 1.0)).astype(np.float32)[..., None]
        self.plate = self.plate * (1 - a) + f * a
        self.known = np.maximum(self.known, learn.astype(np.float32))
    def get(self):
        if self.known.min() > 0: return self.plate
        hole = (self.known == 0).astype(np.uint8) * 255
        return cv2.inpaint(np.clip(self.plate, 0, 255).astype(np.uint8), hole, 7, cv2.INPAINT_TELEA).astype(np.float32)
