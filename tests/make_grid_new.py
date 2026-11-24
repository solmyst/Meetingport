"""Contact sheet of the clone + kamehameha demo MP4s -> demo/demo_new_grid.jpg"""
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
for k, ts in [("clone", [1.9, 2.15, 2.3, 2.45, 2.7, 3.3, 4.5, 5.6]),
              ("kamehameha", [1.9, 2.25, 2.6, 2.85, 2.95, 3.2, 3.55, 3.8])]:
    fr = frames(f"demo/demo_{k}.mp4")      # output starts at clip t=1.0s (after the 1s background capture)
    rows.append(np.hstack([tile(fr[min(len(fr) - 1, int((t - 1) * 30))], f"{k} {t:.2f}s") for t in ts]))
cv2.imwrite("demo/demo_new_grid.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("ok")
