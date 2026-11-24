"""End-to-end, headless: synthetic room + real person photo, forced and real-gesture triggers."""
import os, sys, json, subprocess, pytest, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("mediapipe"); import cv2
CLIPS = os.path.join(ROOT, "demo", "clips")

@pytest.fixture(scope="session", autouse=True)
def clips():
    if not os.path.exists(os.path.join(CLIPS, "person_palm.mp4")):
        subprocess.check_call([sys.executable, "tests/make_clips.py"], cwd=ROOT)

def run(tmp, src, force, extra=()):
    out, log = str(tmp / "o.mp4"), str(tmp / "e.json")
    subprocess.run([sys.executable, "meetingport.py", "--source", os.path.join(CLIPS, src), "--plate-seconds", "1",
                    "--force", force, "--no-preview", "--no-sound", "--out", out, "--log-json", log, *extra], cwd=ROOT, check=True, capture_output=True)
    cap = cv2.VideoCapture(out); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f.astype(np.float32))
    return fr, json.load(open(log))

def empty_room(src="person.mp4"):
    cap = cv2.VideoCapture(os.path.join(CLIPS, src)); fs = [cap.read()[1].astype(np.float32) for _ in range(20)]
    return np.median(fs, axis=0)   # the clip's own empty-room frames (same codec as the output)

def diff(f, room): return float(np.abs(f - room).mean())

@pytest.mark.parametrize("kind,extra", [("goku", ()), ("naruto", ("--substitution", "none"))])
def test_vanish_and_reappear(tmp_path, kind, extra):
    fr, ev = run(tmp_path, "person.mp4", f"{kind}@2.0,palm@4.4", extra)
    room = empty_room()
    at = lambda t: fr[int(round((t - 1.0) * 30))]
    assert [e["gesture"] for e in ev] == [kind, "palm"]
    assert diff(at(1.8), room) > 8          # person visible
    assert diff(at(4.0), room) < 3          # gone: output is the rebuilt empty room
    assert diff(at(6.5), room) > 8          # back after the reverse effect

def test_real_open_palm_photo_triggers_reappear(tmp_path):
    fr, ev = run(tmp_path, "person_palm.mp4", "goku@2.0")
    assert [e["gesture"] for e in ev] == ["goku", "palm"]
    palm = ev[1]
    assert palm["source"] == "gesture" and 5.4 <= palm["t"] <= 6.0   # palm shown at 4.6s, ~1.0s hold

def test_no_false_triggers_without_gestures(tmp_path):
    _, ev = run(tmp_path, "person.mp4", "")
    assert ev == []                          # pointing-up person for 6s: nothing fires

def test_diff_fallback_segmenter(tmp_path):
    fr, ev = run(tmp_path, "person.mp4", "goku@2.0", ["--segmenter", "diff"])
    room = empty_room()
    assert diff(fr[int((4.0 - 1) * 30)], room) < 3

def test_shadow_clones_persist_then_dismiss(tmp_path):
    fr, ev = run(tmp_path, "person.mp4", "clone@2.0,palm@5.0")
    room = empty_room(); at = lambda t: fr[int(round((t - 1.0) * 30))]
    assert [e["gesture"] for e in ev] == ["clone", "unclone"]
    base = diff(at(1.8), room)                       # just you
    assert diff(at(3.6), room) > base * 1.6          # clones out (persisting, well after the effect ended)
    assert diff(at(4.6), room) > base * 1.6
    assert abs(diff(at(6.5), room) - base) < base * 0.3   # dismissed: back to just you, not vanished

def test_kamehameha_one_shot_keeps_you(tmp_path):
    fr, ev = run(tmp_path, "person.mp4", "kamehameha@2.0,palm@5.0")
    room = empty_room(); at = lambda t: fr[int(round((t - 1.0) * 30))]
    assert [e["gesture"] for e in ev] == ["kamehameha"]          # palm ignored: you never left
    assert at(3.0).mean() > at(1.8).mean() + 15                   # beam lights up the frame
    assert abs(diff(at(4.5), room) - diff(at(1.8), room)) < 4     # afterwards: you, as before

def test_gojo_void_keeps_you_then_dismisses(tmp_path):
    """Gojo opens the Infinite Void under the person (they stay) and palm collapses it back to the room."""
    fr, ev = run(tmp_path, "person.mp4", "gojo@2.0,palm@5.0")
    room = empty_room(); at = lambda t: fr[int(round((t - 1.0) * 30))]
    assert [e["gesture"] for e in ev] == ["gojo", "unvoid"]
    base = diff(at(1.8), room)                       # just you, in the room
    assert base > 8
    # Fully open (~t=3.4, after the 1.1s expansion): you are still there, but the room is gone.
    mid = at(3.6)
    assert diff(mid, room) > base                    # background is no longer the empty room
    # The person (bright cutout) is still in the middle of the frame; a vanish would be almost the room.
    assert mid[200:400, 350:600].mean() > 40
    # After the collapse: back to you in the room, not vanished.
    assert abs(diff(at(6.5), room) - base) < base * 0.4
