"""Clean-plate robustness: people/shadows never baked into the empty room; vanish ends on the pure plate."""
import os, sys, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mport.perception import PlateCapture, PlateBuilder
from mport.effects import Effect

H, W = 90, 160
ROOM = np.full((H, W, 3), 120, np.float32); ROOM[:, :40] = 60

def person(x0):
    f = ROOM.copy(); m = np.zeros((H, W), np.float32); m[20:, x0:x0 + 30] = 1; f[m > 0] = 220; return f, m

def test_capture_ignores_person_in_most_frames():
    c = PlateCapture((H, W, 3), 20, min_clean=3, dilate_px=9)
    for i in range(40):
        if c.done(): break
        f, m = person(60) if i % 4 else (ROOM.copy(), np.zeros((H, W), np.float32))   # in frame 75% of the time
        c.add(f, m)
    plate, known = c.result()
    assert known.min() == 1 and np.abs(plate - ROOM).max() < 1e-3   # a median would have baked the person in

def test_capture_refuses_while_person_in_frame():
    c = PlateCapture((H, W, 3), 10, min_clean=5, dilate_px=9)
    for _ in range(200): c.add(*person(60))
    assert c.person_in_frame() and not c.done() and c.frames == 0   # frames with you are rejected outright
    for _ in range(10): c.add(ROOM.copy(), np.zeros((H, W), np.float32))
    assert not c.person_in_frame() and c.done()
    assert np.abs(c.result()[0] - ROOM).max() < 1e-3

def small_thing():   # e.g. a hand at the edge: < 2% of the frame, so the frame counts, minus that spot
    f = ROOM.copy(); m = np.zeros((H, W), np.float32); m[60:70, 70:80] = 1; f[m > 0] = 220; return f, m

def test_capture_masks_small_intrusions_and_gives_up_on_never_seen_pixels():
    c = PlateCapture((H, W, 3), 5, min_clean=2, dilate_px=9, max_factor=2)
    while not c.done(): c.add(*small_thing())
    plate, known = c.result(); assert known[65, 75] == 0 and known[5, 5] == 1
    pb = PlateBuilder((H, W, 3)); pb.seed_capture(c)
    assert abs(pb.get()[65, 75] - 220).max() > 50          # hole inpainted from the room, not the intruder

def test_update_does_not_learn_shadow_but_tracks_drift():
    pb = PlateBuilder((H, W, 3)); pb.plate = ROOM.copy(); pb.known[:] = 1
    shadow = ROOM.copy(); shadow[:, 100:130] *= 0.5        # your cast shadow on the wall, outside the person mask
    for _ in range(50): pb.update(shadow, np.zeros((H, W), np.float32), rate=0.2)
    assert np.abs(pb.plate[:, 100:130] - ROOM[:, 100:130]).max() < 1e-3
    for _ in range(50): pb.update(ROOM + 10, np.zeros((H, W), np.float32), rate=0.2)
    assert np.abs(pb.plate[:, :100] - (ROOM[:, :100] + 10)).max() < 0.5   # slow lighting drift still learned

@pytest.mark.parametrize("kind", ["goku", "naruto"])
def test_vanish_ends_on_pure_plate(kind):
    f, m = person(60); m = np.clip(m + 0.04, 0, 1)                      # soft residue everywhere
    e = Effect(kind, m, f.shape)
    out = e.render(f, m, ROOM, e.duration)
    assert np.abs(out - ROOM).max() < 1e-3
    r = Effect(kind, m, f.shape, reverse=True)                          # first reappear frame too
    assert np.abs(r.render(f, m, ROOM, 0.0) - ROOM).max() < 1e-3
