"""Kawarimi log placement, config parsing, procedural asset, and compositing."""
import json, os, sys, subprocess
import numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cv2 = pytest.importorskip("cv2")
from mport.substitution import (
    LOG_ASPECT, LOG_HEIGHT_FRAC, composite_substitution, generate_procedural_log,
    load_substitution_image, log_placement, person_bbox, prepare_substitution,
)
from mport.app import Meetingport, normalize_substitution, parse
from mport.effects import Effect, naruto_visibility

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "demo", "clips")


# ---------- placement math ----------
def test_placement_uses_bbox_bottom_center():
    bbox = (100, 50, 200, 250)                                # 100 x 200 person
    p = log_placement(bbox, (400, 400), scale=1.0)
    assert p["cx"] == 150
    assert p["bottom"] == 250                                 # not clipped by the frame
    assert abs(p["height"] - 200 * LOG_HEIGHT_FRAC) < 1e-6
    assert abs(p["width"] - p["height"] * LOG_ASPECT) < 1e-6
    assert p["tilt_deg"] != 0


def test_scale_multiplies_size_not_position():
    bbox = (100, 50, 200, 250)
    a = log_placement(bbox, (400, 400), scale=1.0)
    b = log_placement(bbox, (400, 400), scale=2.0)
    assert abs(b["height"] - 2 * a["height"]) < 1e-6
    assert abs(b["width"] - 2 * a["width"]) < 1e-6
    assert b["cx"] == a["cx"] and b["bottom"] == a["bottom"]


def test_empty_bbox_fallback_is_on_screen():
    p = log_placement(None, (180, 320))
    assert 0 < p["cx"] < 320
    assert 0 < p["bottom"] <= 180
    assert p["height"] > 8 and p["width"] > 4


def test_huge_scale_is_clamped_to_the_frame():
    p = log_placement((10, 10, 300, 170), (180, 320), scale=10)
    assert p["height"] <= 180 * 0.85 + 1e-6


def test_bbox_flush_with_frame_lifts_the_base():
    """A person cut off at the bottom of the frame shouldn't leave a clipped log."""
    h, w = 180, 320
    p = log_placement((80, 20, 160, h), (h, w), scale=1.0)
    assert p["bottom"] < h - 1
    assert p["bottom"] > h * 0.7


def test_person_bbox_empty_is_none():
    assert person_bbox(np.zeros((10, 10), np.float32)) is None
    assert person_bbox(None) is None
    m = np.zeros((20, 30), np.float32); m[5:15, 8:22] = 1
    assert person_bbox(m) == (8, 5, 21, 14)


# ---------- config / CLI ----------
def test_parse_defaults_to_log():
    cfg = parse([])
    assert cfg["naruto_substitution"] == "log"
    assert cfg["substitution_scale"] == 1.0
    assert cfg["substitution_image"] == ""


def test_parse_cli_overrides_substitution():
    cfg = parse(["--substitution", "none"])
    assert cfg["naruto_substitution"] == "none"


def test_parse_config_file_and_cli_override(tmp_path):
    cfgp = tmp_path / "c.json"
    cfgp.write_text(json.dumps({
        "naruto_substitution": "image",
        "substitution_image": "props/log.png",
        "substitution_scale": 1.25,
    }))
    cfg = parse(["--config", str(cfgp)])
    assert cfg["naruto_substitution"] == "image"
    assert cfg["substitution_image"] == "props/log.png"
    assert cfg["substitution_scale"] == 1.25
    cfg2 = parse(["--config", str(cfgp), "--substitution", "none"])
    assert cfg2["naruto_substitution"] == "none"
    assert cfg2["substitution_image"] == "props/log.png"


def test_normalize_coerces_bad_values():
    assert normalize_substitution({"naruto_substitution": "LOG"})["naruto_substitution"] == "log"
    assert normalize_substitution({"naruto_substitution": "banana"})["naruto_substitution"] == "log"
    assert normalize_substitution({"substitution_scale": "1.5"})["substitution_scale"] == 1.5
    assert normalize_substitution({"substitution_scale": -2})["substitution_scale"] == 1.0
    assert normalize_substitution({"substitution_scale": "nope"})["substitution_scale"] == 1.0
    assert normalize_substitution({"substitution_image": None})["substitution_image"] == ""


# ---------- asset + composite ----------
def test_procedural_log_has_alpha_bark_and_a_pale_cut_end():
    im = generate_procedural_log(200, 80)
    assert im.shape[2] == 4 and im.dtype == np.uint8
    opaque = im[..., 3] > 40
    assert opaque.mean() > 0.2 and opaque.mean() < 0.9        # not a solid rectangle
    # The cut end at the top of the log is paler than the bark mid-section.
    ys, xs = np.where(opaque)
    y0, y1 = ys.min(), ys.max()
    top = im[y0:y0 + max(4, (y1 - y0) // 6)]
    mid = im[y0 + (y1 - y0) // 3:y0 + (y1 - y0) // 2]
    top_px = top[top[..., 3] > 40][..., :3].mean()
    mid_px = mid[mid[..., 3] > 40][..., :3].mean()
    assert top_px > mid_px + 15


def test_prepare_none_returns_none():
    mask = np.zeros((180, 320), np.float32); mask[40:160, 120:200] = 1
    assert prepare_substitution("none", mask, (180, 320, 3)) is None
    rgba, place = prepare_substitution("log", mask, (180, 320, 3))
    assert rgba.shape[2] == 4 and place["cx"] == 159.5


def test_prepare_missing_image_falls_back_to_the_log():
    mask = np.zeros((180, 320), np.float32); mask[40:160, 120:200] = 1
    rgba, _ = prepare_substitution("image", mask, (180, 320, 3), image_path="no-such-file.png")
    assert rgba.shape[2] == 4


def test_load_and_composite_custom_png(tmp_path):
    # A solid red rectangle with a soft alpha, used as a stand-in "log".
    png = np.zeros((80, 40, 4), np.uint8)
    png[10:70, 8:32] = (0, 0, 220, 255)
    path = tmp_path / "prop.png"
    cv2.imwrite(str(path), png)
    loaded = load_substitution_image(str(path), 100, 50)
    assert loaded is not None and loaded.shape[2] == 4
    assert loaded[..., 2].max() > 200                         # red channel survives the fit
    plate = np.full((180, 320, 3), 180, np.float32)
    place = log_placement((100, 40, 180, 160), plate.shape, scale=1.0)
    out = composite_substitution(plate, loaded, place, alpha=1.0)
    # The prop should have darkened / colored a patch near the bbox center.
    assert np.abs(out - plate).mean() > 1
    # alpha 0 leaves the plate alone
    same = composite_substitution(plate, loaded, place, alpha=0.0)
    assert np.array_equal(same, plate)


def test_composite_adds_a_contact_shadow_and_respects_alpha():
    plate = np.full((180, 320, 3), 200, np.float32)
    mask = np.zeros((180, 320), np.float32); mask[30:160, 110:210] = 1
    rgba, place = prepare_substitution("log", mask, plate.shape)
    out = composite_substitution(plate, rgba, place, alpha=1.0)
    # Region around the planted base is darker than the raw plate (the shadow).
    by = int(place["bottom"])
    cx = int(place["cx"])
    band = out[max(0, by - 6):min(180, by + 4), max(0, cx - 20):min(320, cx + 20)]
    # Some of those pixels are the log itself; the mean should still drop.
    assert band.mean() < 200 - 5
    faded = composite_substitution(plate, rgba, place, alpha=0.2)
    full = composite_substitution(plate, rgba, place, alpha=1.0)
    # A faint paste changes the plate less than a solid one.
    assert np.abs(faded - plate).mean() < np.abs(full - plate).mean()


def test_naruto_visibility_fades_the_person_under_the_smoke():
    assert naruto_visibility(0.0) == pytest.approx(1.0)
    assert naruto_visibility(0.20) == pytest.approx(0.0)
    assert 0 < naruto_visibility(0.12) < 1


def test_naruto_effect_shows_the_log_after_the_puff_and_hides_it_on_reappear():
    """The log is on the plate the effect composites over, faded with person visibility."""
    H, W = 180, 320
    y = np.linspace(0, 1, H)[:, None, None]
    room = (np.array([180, 200, 215]) * (1 - 0.25 * y) + np.zeros((H, W, 3))).astype(np.float32)
    room[int(H * .72):] = (60, 90, 120)
    frame = room.copy()
    mask = np.zeros((H, W), np.float32)
    mask[40:H, W // 2 - 36:W // 2 + 36] = 1
    frame[mask > 0] = (70, 110, 200)
    rgba, place = prepare_substitution("log", mask, frame.shape)
    # Forward (vanish): at t=0 the person is visible; at t=1 the log sits on the plate.
    ev = Effect("naruto", mask, frame.shape)
    plate0 = room
    early = ev.render(frame, mask, plate0, 0.0)
    assert np.abs(early[mask > 0] - frame[mask > 0]).mean() < 8
    plate1 = composite_substitution(room, rgba, place, alpha=1.0)
    late = ev.render(frame, mask, plate1, ev.duration)
    # After the puff, output matches the log-on-plate, not the empty room or the person.
    assert np.abs(late - plate1).mean() < 2
    cy, cx = int(place["bottom"] - place["height"] * 0.5), int(place["cx"])
    patch = np.s_[max(0, cy - 25):cy + 25, max(0, cx - 25):cx + 25]
    assert np.abs(late[patch] - room[patch]).mean() > 8
    # Reverse (reappear): start is the log, end is the person and no log.
    rev = Effect("naruto", mask, frame.shape, reverse=True)
    start = rev.render(frame, mask, plate1, 0.0)
    assert np.abs(start - plate1).mean() < 2
    end = rev.render(frame, mask, room, rev.duration)          # log already faded (alpha 0)
    assert np.abs(end[mask > 0] - frame[mask > 0]).mean() < 8
    assert np.abs(end[mask < 0.1] - room[mask < 0.1]).mean() < 8


def test_cycle_substitution_toggles_log_and_none():
    cfg = parse(["--no-gestures", "--segmenter", "diff", "--no-sound", "--substitution", "log"])
    mp = Meetingport(cfg, (180, 320, 3), 30)
    mp.trig.present = False
    mp.sub_bbox = (100, 40, 180, 160)
    assert mp.cycle_substitution() == "none"
    assert mp.cfg["naruto_substitution"] == "none"
    assert mp.substitution is None
    assert mp.cycle_substitution() == "log"
    assert mp.substitution is not None


def _run_app(tmp, src, force, extra=()):
    out, log = str(tmp / "o.mp4"), str(tmp / "e.json")
    subprocess.run(
        [sys.executable, "meetingport.py", "--source", os.path.join(CLIPS, src),
         "--plate-seconds", "1", "--force", force, "--no-preview", "--no-sound",
         "--segmenter", "diff", "--out", out, "--log-json", log, *extra],
        cwd=ROOT, check=True, capture_output=True)
    cap = cv2.VideoCapture(out); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f.astype(np.float32))
    return fr, json.load(open(log))


def empty_room(src="person.mp4"):
    cap = cv2.VideoCapture(os.path.join(CLIPS, src))
    fs = [cap.read()[1].astype(np.float32) for _ in range(20)]
    return np.median(fs, axis=0)


@pytest.mark.skipif(not os.path.exists(os.path.join(CLIPS, "person.mp4")), reason="demo clip missing")
def test_naruto_log_stays_while_vanished_and_clears_on_reappear(tmp_path):
    """Full app loop (diff segmenter, no MediaPipe) with the default log substitution."""
    fr, ev = _run_app(tmp_path, "person.mp4", "naruto@2.0,palm@4.4")
    room = empty_room()
    at = lambda t: fr[int(round((t - 1.0) * 30))]
    assert [e["gesture"] for e in ev] == ["naruto", "palm"]
    assert float(np.abs(at(1.8) - room).mean()) > 8           # person visible
    gone = at(4.0)
    # Not a pure empty room: the log is sitting where the person was.
    assert float(np.abs(gone - room).mean()) > 3
    # The top of the frame (wall / poster) is still the room.
    assert float(np.abs(gone[:80] - room[:80]).mean()) < 3
    # The log lives in the lower-middle, not the far corners.
    mid = gone[200:500, 350:610]
    mid_room = room[200:500, 350:610]
    assert float(np.abs(mid - mid_room).mean()) > 5
    # After reappear: the person is back (not an empty room, and not just a leftover log).
    back = at(6.5)
    assert float(np.abs(back - room).mean()) > 8
