"""Contact sheet of the demo MP4s -> demo/demo_grid.jpg"""
import cv2, numpy as np
def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: return out
        out.append(f)
def tile(f, label):
    f = cv2.resize(f, (320, 180)); cv2.rectangle(f, (0, 0), (190, 20), (0, 0, 0), -1)
    cv2.putText(f, label, (5, 15), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA); return f
rows = []
for k, d in [("goku", 1.0), ("naruto", 1.3), ("gojo", 1.6)]:
    fr = frames(f"demo/demo_{k}.mp4")      # output starts at clip t=1.0s (after the 1s background capture)
    ts = [1.8] + list(np.linspace(2.08, 2 + d * 0.9, 5)) + [3.6, 4.4 + d * 0.85 * 0.45]
    rows.append(np.hstack([tile(fr[min(len(fr) - 1, int((t - 1) * 30))], f"{k} {t:.2f}s") for t in ts]))
fr = frames("demo/demo_real_palm_gesture.mp4")
rows.append(np.hstack([tile(fr[int((t - 1) * 30)], f"real palm {t:.2f}s") for t in [1.5, 2.5, 3.5, 4.7, 5.0, 5.3, 5.65, 6.2]]))
cv2.imwrite("demo/demo_grid.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("ok")
