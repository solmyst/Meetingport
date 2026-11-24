"""Build synthetic 'webcam' clips: a rendered room + a real person (MediaPipe Apache-2.0 sample photo, cut out
with the selfie segmenter) who sways slightly. Optional: a real open-palm hand photo pasted in for a time window."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
W, H, FPS = 960, 540, 30

def room(rng=np.random.default_rng(5)):
    y = np.linspace(0, 1, H)[:, None, None]
    img = (np.array([180, 200, 215]) * (1 - 0.25 * y) + np.zeros((H, W, 3))).astype(np.float32)   # warm wall
    img[int(H * .72):] = np.array([60, 90, 120])                                                    # desk
    cv2.rectangle(img, (60, 70), (230, 260), (90, 60, 40), -1); cv2.rectangle(img, (72, 82), (218, 248), (200, 170, 120), -1)  # frame/poster
    cv2.rectangle(img, (720, 120), (900, 135), (50, 70, 100), -1)                                   # shelf
    for i, c in enumerate([(40, 40, 180), (160, 90, 30), (40, 140, 60), (30, 160, 200)]):
        cv2.rectangle(img, (735 + i * 38, 70), (760 + i * 38, 120), c, -1)                         # books
    cv2.circle(img, (840, 330), 55, (50, 130, 60), -1); cv2.rectangle(img, (815, 370), (865, 390), (40, 60, 120), -1)  # plant
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def person_cutout():
    from mport.perception import MPSegmenter
    im = cv2.imread(os.path.join(A, "pointing_up.jpg")); im = cv2.resize(im, (int(im.shape[1] * 470 / im.shape[0]), 470))
    m = MPSegmenter(smooth=0)(im, 0)
    return im, cv2.GaussianBlur((m > 0.5).astype(np.float32), (5, 5), 0)

def hand_cutout():
    im = cv2.imread(os.path.join(A, "right_hands.jpg"))
    h = im[:, im.shape[1] // 2:]                       # right half: one open palm
    m = (cv2.cvtColor(h, cv2.COLOR_BGR2GRAY) < 225).astype(np.float32)
    m = cv2.GaussianBlur(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5))), (5, 5), 0)
    s = 300 / h.shape[0]; return cv2.resize(h, None, fx=s, fy=s), cv2.resize(m, None, fx=s, fy=s)

def paste(dst, src, m, x, y):
    h, w = src.shape[:2]; x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32); mm = m[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
    dst[y0:y1, x0:x1] = (dst[y0:y1, x0:x1] * (1 - mm) + s * mm).astype(np.uint8)

def make(path, seconds=7.0, empty=1.0, palm=None):
    bg = room(); p, pm = person_cutout(); hnd, hm = hand_cutout()
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)); rng = np.random.default_rng(1)
    for i in range(int(seconds * FPS)):
        t = i / FPS; f = bg.copy()
        if t >= empty: paste(f, p, pm, W // 2 - p.shape[1] // 2 + int(8 * np.sin(t * 2.1)), H - p.shape[0] + int(3 * np.sin(t * 3.3)))
        if palm and palm[0] <= t < palm[1]: paste(f, hnd, hm, 120, 150)
        f = np.clip(f.astype(np.int16) + rng.integers(-3, 4, f.shape), 0, 255).astype(np.uint8)  # sensor noise
        vw.write(f)
    vw.release(); cv2.imwrite(path.replace(".mp4", "_room.png"), bg)

if __name__ == "__main__":
    os.makedirs("demo/clips", exist_ok=True)
    make("demo/clips/person.mp4", 7.0)
    make("demo/clips/person_palm.mp4", 8.0, palm=(4.6, 6.6))   # real open-palm photo appears 4.6s..6.6s
    print("ok")
