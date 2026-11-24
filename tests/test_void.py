"""Infinite Void renderer + gojo effect: the person stays, the background is replaced, captions optional."""
import os, sys, time, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cv2 = pytest.importorskip("cv2")
from mport.effects import Effect
from mport.void import InfiniteVoid, get_void

H, W = 180, 320

def _scene():
    """A warm room and a solid person rectangle (skin-toned torso + darker hair)."""
    y = np.linspace(0, 1, H)[:, None, None]
    room = (np.array([180, 200, 215]) * (1 - 0.25 * y) + np.zeros((H, W, 3))).astype(np.float32)
    room[int(H * .72):] = (60, 90, 120)
    frame = room.copy()
    mask = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = W // 2 - 36, 40, W // 2 + 36, H
    mask[y0:y1, x0:x1] = 1
    frame[y0:y1, x0:x1] = (70, 110, 200)                     # shirt (fills the whole silhouette)
    frame[y0:y0 + 28, x0:x1] = (40, 40, 40)                  # hair
    frame[y0 + 28:y0 + 55, x0 + 6:x1 - 6] = (140, 170, 220)  # face
    return frame, mask, room

def _interior(mask, frac=0.22):
    """Central part of the silhouette, away from the rim-light band."""
    ys, xs = np.where(mask > 0.5)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    dx, dy = int((x1 - x0) * frac), int((y1 - y0) * frac)
    m = np.zeros(mask.shape, dtype=bool)
    m[y0 + dy:max(y0 + dy + 1, y1 - dy), x0 + dx:max(x0 + dx + 1, x1 - dx)] = mask[y0 + dy:max(y0 + dy + 1, y1 - dy), x0 + dx:max(x0 + dx + 1, x1 - dx)] > 0.5
    return m

def test_void_keeps_person_visible_and_replaces_background():
    frame, mask, room = _scene()
    e = Effect("gojo", mask, frame.shape, anchor=(W * 0.35, H * 0.55))
    # Fully open
    out = e.render(frame, mask, room, e.duration)
    person = _interior(mask)
    bg = mask < 0.15
    # Interior person pixels stay close to the original frame (fully visible, not dissolved).
    assert person.any() and np.abs(out[person] - frame[person]).mean() < 8
    # Background is no longer the room: it should be dark (the void).
    assert out[bg].mean() < 70
    assert np.abs(out[bg] - room[bg]).mean() > 40
    # And we did not end on the pure plate (the old vanish contract).
    assert np.abs(out - room).mean() > 20

def test_void_opening_is_a_partial_sphere_then_full():
    frame, mask, room = _scene()
    e = Effect("gojo", mask, frame.shape, anchor=(40, 90))
    early = e.render(frame, mask, room, 0.15 * e.duration)
    full = e.render(frame, mask, room, e.duration)
    bg = mask < 0.15
    # Early on, most of the far background is still the room; fully open, it's the void.
    assert np.abs(early[bg] - room[bg]).mean() < np.abs(full[bg] - room[bg]).mean()
    # The person is visible in both (compare the interior, away from the rim light).
    person = _interior(mask)
    assert person.any() and np.abs(early[person] - frame[person]).mean() < 8
    assert np.abs(full[person] - frame[person]).mean() < 8

def test_void_close_returns_the_room_behind_the_person():
    frame, mask, room = _scene()
    e = Effect("gojo", mask, frame.shape, reverse=True, anchor=(W * 0.5, H * 0.6))
    # t=0 of a reverse effect is fully open (render flips t to 1).
    open_ = e.render(frame, mask, room, 0.0)
    closed = e.render(frame, mask, room, e.duration)
    bg = mask < 0.15
    person = _interior(mask)
    assert open_[bg].mean() < closed[bg].mean()          # void is darker than the room
    assert np.abs(closed[bg] - room[bg]).mean() < 8      # collapsed: room is back
    assert person.any() and np.abs(closed[person] - frame[person]).mean() < 8

def test_caption_flag_off_has_no_text_block():
    frame, mask, room = _scene()
    fx = InfiniteVoid(frame.shape, seed=3)
    with_ = fx.render(frame, mask, 1.0, 1.5, origin=(80, 90), caption=True)
    without = fx.render(frame, mask, 1.0, 1.5, origin=(80, 90), caption=False)
    # Caption lives near the bottom; those rows should differ when the flag is on.
    assert np.abs(with_[-40:] - without[-40:]).mean() > 0.5
    # And the rest of the frame (person + void) should be essentially identical.
    assert np.abs(with_[:-50] - without[:-50]).mean() < 0.05

def test_get_void_reuses_precomputed_instance():
    a = get_void((H, W, 3)); b = get_void((H, W, 3))
    assert a is b

def test_void_render_is_realtime_at_720p():
    """Aim well under ~15 ms/frame at 1280x720 on CPU (warmup excluded)."""
    h, w = 720, 1280
    y = np.linspace(0, 1, h)[:, None, None]
    room = (np.array([180, 200, 215], np.float32) * (1 - 0.2 * y) + np.zeros((h, w, 3))).astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    mask[140:h, 490:790] = 1
    frame = room.copy(); frame[mask > 0] = (80, 120, 200)
    fx = InfiniteVoid((h, w, 3), seed=11)
    # warmup (JIT-ish caches, first remap)
    for i in range(3):
        fx.render(frame, mask, 1.0, 0.4 * i, origin=(700, 420), caption=True)
    times = []
    for i in range(40):
        t0 = time.perf_counter()
        fx.render(frame, mask, 1.0, 0.4 + i / 30.0, origin=(700, 420), caption=True)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean, p95 = float(np.mean(times)), float(np.percentile(times, 95))
    print(f"void 1280x720: mean={mean:.2f}ms p95={p95:.2f}ms  n={len(times)}")
    # Loose CI bound so a busy box doesn't flake; the measured numbers are reported either way.
    assert mean < 25.0, f"mean {mean:.2f}ms is too slow for real-time"
