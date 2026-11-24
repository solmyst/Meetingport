"""Meetingport: 'teleport' out of a video call. Captures an empty-room background plate,
then on trigger dissolves you (Goku-style vertical streaks) into that plate and keeps
sending the empty room to a virtual camera. Works with a webcam or a video file."""
import argparse, time, cv2, numpy as np

def person_mask(frame, plate, thr=35):
    d = cv2.absdiff(cv2.GaussianBlur(frame,(7,7),0), cv2.GaussianBlur(plate,(7,7),0)).max(axis=2)
    m = (d > thr).astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8)), None, iterations=2)
    return cv2.GaussianBlur(m,(21,21),0).astype(np.float32)/255

def teleport_frame(frame, plate, t, rng):
    """t in [0,1]: 0 = you, 1 = gone. Streak dissolve + flash."""
    h, w = frame.shape[:2]
    m = person_mask(frame, plate)[..., None]
    cols = (rng.random(w) < t**0.7).astype(np.float32)[None, :, None]   # vertical streaks
    gone = np.clip(cols + max(0, t*1.4-0.4), 0, 1)
    out = frame*(1-m*gone) + plate*(m*gone)
    streak = np.zeros_like(out); streak[..., :] = (255,240,200)
    lines = (rng.random(w) < 0.15*np.sin(np.pi*t)).astype(np.float32)[None,:,None]
    out = out*(1-lines*m) + streak*(lines*m)
    flash = max(0, 1-abs(t-0.85)*8)*0.6
    return np.clip(out*(1-flash)+255*flash, 0, 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--plate-seconds", type=float, default=3, help="seconds of empty room to average")
    ap.add_argument("--plate-image", help="use this image as background plate instead")
    ap.add_argument("--trigger-at", type=float, help="auto-teleport at N seconds (headless/testing)")
    ap.add_argument("--duration", type=float, default=1.5, help="effect length (s)")
    ap.add_argument("--out", help="write result to this video file")
    ap.add_argument("--virtualcam", action="store_true", help="send to OBS/v4l2 virtual camera (pip install pyvirtualcam)")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, f = cap.read(); assert ok, "cannot read source"
    h, w = f.shape[:2]
    if a.plate_image: plate = cv2.resize(cv2.imread(a.plate_image), (w, h)).astype(np.float32)
    else:
        print(f"Step out of frame: capturing background for {a.plate_seconds}s"); frames=[f]
        for _ in range(int(a.plate_seconds*fps)):
            ok, f = cap.read()
            if ok: frames.append(f)
        plate = np.median(np.stack(frames), axis=0).astype(np.float32)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if a.out else None
    cam = None
    if a.virtualcam:
        import pyvirtualcam; cam = pyvirtualcam.Camera(w, h, fps); print("virtual cam:", cam.device)
    rng = np.random.default_rng(0); start = None; n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        n += 1; now = n/fps
        if a.trigger_at is not None and start is None and now >= a.trigger_at: start = now; print(f"teleport at {now:.1f}s")
        if start is None: out = f
        elif now-start < a.duration: out = teleport_frame(f.astype(np.float32), plate, (now-start)/a.duration, rng)
        else: out = plate.astype(np.uint8)          # empty room forever: you've left
        if vw: vw.write(out)
        if cam: cam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); cam.sleep_until_next_frame()
        if not a.headless:
            cv2.imshow("Meetingport (T=teleport, R=return, Q=quit)", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('t') and start is None: start = now
            elif k == ord('r'): start = None
            elif k == ord('q'): break
    if vw: vw.release()
    print("frames:", n)

if __name__ == "__main__": main()
