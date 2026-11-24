"""Snap-to-dust: gesture state machine, audio onset, mute commands, effect determinism."""
import math, os, sys, json, subprocess
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mport import gestures as G
from mport.audio import SnapOnsetDetector
from mport.effects import Effect, DURATION
from mport.mute import (
    MicController, draw_muted_badge, macos_get_input_volume_cmd, macos_set_input_volume_cmd,
    macos_hotkey_cmd, linux_get_mute_cmd, linux_set_mute_cmd, windows_hotkey_sendkeys,
    parse_input_volume, parse_pactl_mute, normalize_app,
)
from mport.app import parse, normalize_snap, snap_gate_for, snap_watcher_for, DEFAULTS

cv2 = pytest.importorskip("cv2")

# Reuse the same synthetic-hand builder as test_gestures.py
from tests.test_gestures import hand, FACE, run as run_machine


def _xform(pts, cx, cy, s, rot=0.0):
    c, sn = math.cos(rot), math.sin(rot)
    return [(cx + s * (x * c - y * sn), cy + s * (x * sn + y * c)) for x, y in pts]


def snap_hand(curl=0.4, thumb_on_middle=True, cx=0.6, cy=0.5, s=0.1, rot=0.0,
              thumb_hold=None):
    """Upright hand. curl 0 = middle extended up, 1 = middle tip near the palm.
    thumb_on_middle pins the thumb tip to the (current) middle tip (a pinch).
    thumb_hold, if given, is a local-space point the thumb tip stays on (after a pinch)."""
    # Local coords copied from tests.test_gestures.hand
    pts = [None] * 21
    pts[0] = (0.0, 0.0)
    bx = {"index": -0.35, "middle": -0.1, "ring": 0.12, "pinky": 0.32}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for f, i in base.items():
        x = bx[f]; pts[i] = (x, -1.0)
        if f == "index":
            pts[i + 1] = (x, -1.4); pts[i + 2] = (x, -1.65); pts[i + 3] = (x, -1.9)
        elif f == "middle":
            # lerp extended -> curled
            ext = [(x, -1.4), (x, -1.65), (x, -1.9)]
            cur = [(x, -1.3), (x, -1.05), (x, -0.8)]
            for k in range(3):
                pts[i + 1 + k] = (ext[k][0] + (cur[k][0] - ext[k][0]) * curl,
                                  ext[k][1] + (cur[k][1] - ext[k][1]) * curl)
        else:
            pts[i + 1] = (x, -1.3); pts[i + 2] = (x, -1.05); pts[i + 3] = (x, -0.8)
    # default thumb (extended sideways)
    pts[1:5] = [(-0.3, -0.2), (-0.6, -0.4), (-0.85, -0.6), (-1.05, -0.8)]
    mid_local = pts[12]
    if thumb_on_middle:
        pts[4] = mid_local
    elif thumb_hold is not None:
        pts[4] = thumb_hold
    return _xform(pts, cx, cy, s, rot)


def pinch_pt(curl=0.4, **kw):
    h = snap_hand(curl=curl, thumb_on_middle=True, **kw)
    # local middle tip of the pinch, so the flick can keep the thumb there
    # Recover local tip: invert is messy; just return the world hand and remember curl.
    return h


# ---------- pinch classifier ----------
def test_pinch_when_thumb_on_middle():
    h = snap_hand(curl=0.4, thumb_on_middle=True)
    assert G.is_snap_pinch(h)
    assert G.is_snap_ready(h)
    assert G.classify([h], *FACE)[0] is None          # not a held Meetingport sign


def test_open_palm_is_not_a_pinch():
    h = hand()
    assert not G.is_snap_pinch(h)
    assert not G.is_snap_ready(h)
    assert G.classify([h], *FACE)[0] == G.PALM


def test_loose_thumb_middle_touch_is_not_ready():
    """Incidental proximity (talking hands) must not arm a snap."""
    h = snap_hand(curl=0.4, thumb_on_middle=False)
    # Thumb is out to the side by default: not a pinch at all.
    assert not G.is_snap_pinch(h)
    assert not G.is_snap_ready(h)
    # A loose near-miss (thumb close but outside the tight frac) is also rejected.
    pts = list(h)
    mid = pts[12]
    s = G.hand_size(pts)
    pts[4] = (mid[0] + 0.22 * s, mid[1])
    assert not G.is_snap_pinch(pts, frac=0.18)
    assert not G.is_snap_ready(pts)


def test_fist_may_look_close_but_is_not_a_snap_motion():
    """A static fist must not complete a snap (no pinch-then-flick)."""
    w = G.SnapWatcher()
    h = hand(ext=(), thumb=False)
    fired = [w.update([h], i / 30)[0] for i in range(20)]
    assert not any(fired)


# ---------- SnapWatcher motion ----------
def _play(seq, fps=30, watcher=None, hands_seq=None):
    """seq is one hand per frame, or None. hands_seq overrides with a full hands list per frame."""
    w = watcher or G.SnapWatcher()
    hits = []
    for i, item in enumerate(seq if hands_seq is None else hands_seq):
        if hands_seq is None:
            hands = [item] if item is not None else []
        else:
            hands = item
        fired, anc = w.update(hands, i / fps)
        if fired:
            hits.append((round(i / fps, 3), anc))
    return hits, w


def _snap_sequence(fps=30, pinch_frames=6, flick_curls=(0.55, 0.7, 0.85, 0.95, 1.0)):
    """Hold a snap-ready pinch, then flick the middle finger toward the palm. Thumb stays put."""
    x, curl0 = -0.1, 0.4
    hold = (x, -1.9 + (-0.8 - (-1.9)) * curl0)
    seq = [snap_hand(curl=0.4, thumb_on_middle=True) for _ in range(pinch_frames)]
    for c in flick_curls:
        seq.append(snap_hand(curl=c, thumb_on_middle=False, thumb_hold=hold))
    return seq


def test_snap_motion_fires():
    hits, _ = _play(_snap_sequence())
    assert len(hits) == 1
    # Pinch dwell ~0.2 s, then a quick flick; should complete inside the flick window.
    assert 0.15 <= hits[0][0] <= 0.45


def test_brief_pinch_without_dwell_does_not_fire():
    """Accidental thumb-middle taps while talking: too short to arm."""
    x, curl0 = -0.1, 0.4
    hold = (x, -1.9 + (-0.8 - (-1.9)) * curl0)
    seq = [snap_hand(curl=0.4, thumb_on_middle=True) for _ in range(2)]   # ~0.067 s < min_pinch
    seq += [snap_hand(curl=c, thumb_on_middle=False, thumb_hold=hold) for c in (0.7, 0.85, 0.95, 1.0)]
    hits, _ = _play(seq)
    assert hits == []


def test_pinch_held_without_flick_does_not_fire():
    seq = [snap_hand(curl=0.4, thumb_on_middle=True) for _ in range(30)]
    hits, _ = _play(seq)
    assert hits == []


def test_flick_too_slow_does_not_fire():
    # pinch, then wait past the flick window, then flick
    x, curl0 = -0.1, 0.4
    hold = (x, -1.9 + (-0.8 - (-1.9)) * curl0)
    seq = [snap_hand(curl=0.4, thumb_on_middle=True) for _ in range(6)]
    seq += [None] * 20                                          # 0.67 s gap
    seq += [snap_hand(curl=c, thumb_on_middle=False, thumb_hold=hold) for c in (0.7, 0.85, 0.95)]
    hits, _ = _play(seq)
    assert hits == []


def test_unclenching_is_not_a_snap():
    """Middle finger moving *away* from the palm (opening) must not fire."""
    x, curl0 = -0.1, 0.5
    hold = (x, -1.9 + (-0.8 - (-1.9)) * curl0)
    seq = [snap_hand(curl=0.5, thumb_on_middle=True) for _ in range(6)]
    seq += [snap_hand(curl=c, thumb_on_middle=False, thumb_hold=hold) for c in (0.35, 0.2, 0.05, 0.0)]
    hits, _ = _play(seq)
    assert hits == []


def test_waving_wrist_during_pinch_does_not_fire():
    """Talking gesticulation: the wrist moves too much for a deliberate snap."""
    x, curl0 = -0.1, 0.4
    hold = (x, -1.9 + (-0.8 - (-1.9)) * curl0)
    seq = []
    for i in range(6):
        seq.append(snap_hand(curl=0.4, thumb_on_middle=True, cx=0.55 + 0.04 * i))
    seq += [snap_hand(curl=c, thumb_on_middle=False, thumb_hold=hold, cx=0.55 + 0.04 * 6)
            for c in (0.7, 0.85, 0.95, 1.0)]
    hits, _ = _play(seq)
    assert hits == []


def test_talking_hand_sequences_never_snap():
    """Synthetic 'casual talking' frames: brief near-pinches, pointing, palms, no deliberate flick."""
    w = G.SnapWatcher()
    rng = __import__("random").Random(0)
    hits = []
    for i in range(180):                                      # 6 s at 30 fps
        kind = rng.choice(["point", "palm", "near", "fist", "wave"] * 2 + ["none"])
        if kind == "point":
            h = hand(ext=("index",), thumb=False, cx=0.5 + 0.05 * math.sin(i / 7), cy=0.55)
        elif kind == "palm":
            h = hand(cx=0.55, cy=0.6 + 0.03 * math.sin(i / 5))
        elif kind == "near":
            # Thumb briefly near middle without a held snap-ready pose / flick.
            h = snap_hand(curl=0.35, thumb_on_middle=True, cx=0.5 + 0.08 * math.sin(i / 4))
            if i % 5 != 0:
                h = snap_hand(curl=0.2, thumb_on_middle=False, cx=0.5 + 0.08 * math.sin(i / 4))
        elif kind == "fist":
            h = hand(ext=(), thumb=False, cx=0.5, cy=0.65)
        elif kind == "wave":
            h = hand(ext=("index", "middle", "ring", "pinky"), cx=0.4 + 0.15 * math.sin(i / 3), cy=0.5)
        else:
            h = None
        fired, _ = w.update([h] if h is not None else [], i / 30)
        if fired:
            hits.append(i / 30)
    assert hits == []


def test_two_hands_block_snap_even_with_a_flick():
    """Naruto / clone / kame forming: never complete a snap while both hands are visible."""
    x, curl0 = -0.1, 0.4
    hold = (x, -1.9 + (-0.8 - (-1.9)) * curl0)
    other = hand(ext=("index", "middle"), thumb=False, cx=0.35, cy=0.75)
    hands_seq = []
    for _ in range(6):
        hands_seq.append([snap_hand(curl=0.4, thumb_on_middle=True), other])
    for c in (0.7, 0.85, 0.95, 1.0):
        hands_seq.append([snap_hand(curl=c, thumb_on_middle=False, thumb_hold=hold), other])
    hits, _ = _play(None, hands_seq=hands_seq)
    assert hits == []
    # Same motion with one hand still fires.
    hits1, _ = _play(_snap_sequence())
    assert len(hits1) == 1


def test_snap_does_not_collide_with_named_signs():
    for builder, name in (
        (lambda: hand(), "palm"),
        (lambda: hand(ext=("index", "middle"), thumb=False, crossed=True, cy=0.8), "gojo"),
        (lambda: hand(ext=("index",), thumb=False), "point"),
    ):
        w = G.SnapWatcher()
        h = builder()
        assert not any(w.update([h], i / 30)[0] for i in range(12)), name


def test_snap_watcher_for_reads_config():
    cfg = normalize_snap({"snap_min_pinch": 0.2, "snap_one_hand_only": False})
    w = snap_watcher_for(cfg)
    assert w.min_pinch == 0.2 and w.one_hand_only is False
    assert abs(w.pinch_frac - DEFAULTS["snap_pinch_frac"]) < 1e-9


# ---------- TriggerMachine: snap is a vanish, instant, respects cooldown / void / clones ----------
def test_snap_fires_immediately_and_vanishes():
    f, m = run_machine([G.SNAP])
    assert [g for _, g in f] == [G.SNAP] and not m.present and m.state == "gone"


def test_snap_blocked_while_cloned_or_voided():
    f, m = run_machine([G.CLONE] * 25 + [None] * 70 + [G.SNAP] * 5)
    assert [g for _, g in f] == [G.CLONE] and m.state == "cloned"
    f, m = run_machine([G.GOJO] * 25 + [None] * 70 + [G.SNAP] * 5)
    assert [g for _, g in f] == [G.GOJO] and m.state == "void" and m.present


def test_snap_while_gone_brings_you_back_after_cooldown():
    f, m = run_machine([G.SNAP] + [None] * 70 + [G.SNAP])
    assert [g for _, g in f] == [G.SNAP, G.PALM] and m.state == "present"


def test_snap_cooldown_blocks_instant_reappear():
    f, _ = run_machine([G.SNAP, G.PALM])          # palm on the very next frame: still in cooldown
    assert [g for _, g in f] == [G.SNAP]


def test_palm_reappear_after_snap():
    f, m = run_machine([G.SNAP] + [None] * 70 + [G.PALM] * 35)
    assert [g for _, g in f] == [G.SNAP, G.PALM] and m.state == "present"


# ---------- SnapGate fusion ----------
def test_snap_gate_modes():
    def run(mode, events):
        g = G.SnapGate(mode=mode, window=0.30)
        hits = []
        for ev in events:
            if g.update(ev.get("g", False), ev.get("a", False), ev["t"]):
                hits.append(ev["t"])
        return hits

    # gesture-only ignores audio
    assert run("gesture", [{"t": 0.1, "a": True}, {"t": 0.2, "g": True}]) == [0.2]
    # audio-only ignores gesture
    assert run("audio", [{"t": 0.1, "g": True}, {"t": 0.2, "a": True}]) == [0.2]
    # either fires on the first of the two
    assert run("either", [{"t": 0.1, "a": True}, {"t": 0.2, "g": True}]) == [0.1]
    # both: needs the pair within 300 ms
    assert run("both", [{"t": 0.10, "g": True}, {"t": 0.25, "a": True}]) == [0.25]
    assert run("both", [{"t": 0.10, "a": True}, {"t": 0.25, "g": True}]) == [0.25]
    assert run("both", [{"t": 0.10, "g": True}, {"t": 0.50, "a": True}]) == []   # 400 ms apart
    assert run("both", [{"t": 0.10, "g": True}]) == []
    assert run("both", [{"t": 0.10, "a": True}]) == []


# ---------- Audio onset ----------
def _click(sr=16000, t0=0.30, dur=0.008, total=1.0, seed=3):
    rng = np.random.default_rng(seed)
    n = int(sr * total)
    x = np.zeros(n, np.float32)
    k = int(dur * sr)
    burst = rng.standard_normal(k).astype(np.float32)
    burst = np.diff(burst, prepend=burst[:1])          # high-pass: a bright click
    burst *= np.hanning(k).astype(np.float32)
    burst /= np.max(np.abs(burst)) + 1e-9
    i0 = int(t0 * sr)
    x[i0:i0 + k] = burst
    return x


def _speech(sr=16000, total=1.0, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * total)) / sr
    # Glottal-ish pulse + slow syllables, energy concentrated well below 2 kHz.
    glot = np.clip(np.sin(2 * np.pi * 140 * t), 0, None) ** 3
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    noise = rng.standard_normal(t.size).astype(np.float32)
    # crude one-pole low-pass ~ 800 Hz
    a = 2 * np.pi * 800 / sr
    lp = np.zeros_like(noise)
    acc = 0.0
    for i, v in enumerate(noise):
        acc += a * (v - acc); lp[i] = acc
    x = (0.8 * glot * env + 0.4 * lp * env).astype(np.float32)
    x /= np.max(np.abs(x)) + 1e-9
    return x * 0.4


def _tone(sr=16000, freq=4000, total=1.0):
    t = np.arange(int(sr * total)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_audio_click_detected_speech_and_tone_are_not():
    sr = 16000
    det = SnapOnsetDetector(sr=sr)
    assert det.process(_click(sr)) is True
    det.reset()
    assert det.process(_speech(sr)) is False
    det.reset()
    assert det.process(_tone(sr)) is False
    det.reset()
    assert det.process(np.zeros(sr, np.float32)) is False


def test_audio_click_is_not_double_fired_inside_cooldown():
    sr = 16000
    det = SnapOnsetDetector(sr=sr, cooldown=0.25)
    x = _click(sr, t0=0.20, total=0.8, seed=3)
    # second click 80 ms later, inside cooldown
    y = _click(sr, t0=0.28, total=0.8, seed=5)
    z = np.clip(x + y, -1, 1)
    # process in 40 ms hops and count firings
    n, hops, fires = 0, int(0.04 * sr), 0
    while n + hops <= z.size:
        if det.process(z[n:n + hops]):
            fires += 1
        n += hops
    assert fires == 1


# ---------- Mute command construction ----------
class _Proc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


def test_macos_volume_commands_and_restore():
    assert macos_set_input_volume_cmd(0) == ["osascript", "-e", "set volume input volume 0"]
    assert macos_set_input_volume_cmd(75)[-1].endswith("75")
    assert "input volume of" in macos_get_input_volume_cmd()[-1]
    assert parse_input_volume("75\n") == 75
    assert parse_input_volume("oops") is None

    calls = []
    def runner(cmd, timeout=3):
        calls.append(cmd)
        if "input volume of" in " ".join(cmd):
            return _Proc(0, "64\n")
        return _Proc(0, "")

    c = MicController(method="volume", platform="darwin", runner=runner, log=lambda *a: None)
    assert c.mute() is True and c.muted
    assert any("set volume input volume 0" in " ".join(cmd) for cmd in calls)
    assert c._prev == 64
    assert c.unmute() is True and not c.muted
    assert any("set volume input volume 64" in " ".join(cmd) for cmd in calls)


def test_macos_hotkey_commands_per_app():
    meet = " ".join(macos_hotkey_cmd("meet"))
    assert 'keystroke "d"' in meet and "command down" in meet
    zoom = " ".join(macos_hotkey_cmd("zoom"))
    assert 'keystroke "a"' in zoom and "shift down" in zoom and "command down" in zoom
    teams = " ".join(macos_hotkey_cmd("teams"))
    assert 'keystroke "m"' in teams and "shift down" in teams
    assert normalize_app("Google Meet") == "meet"


def test_linux_pactl_commands_and_restore():
    assert linux_set_mute_cmd(True) == ["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "1"]
    assert linux_set_mute_cmd(False)[-1] == "0"
    assert parse_pactl_mute("Mute: yes\n") is True
    assert parse_pactl_mute("Mute: no\n") is False
    calls = []
    def runner(cmd, timeout=3):
        calls.append(cmd)
        if "get-source-mute" in cmd:
            return _Proc(0, "Mute: no\n")
        return _Proc(0, "")
    c = MicController(method="volume", platform="linux", runner=runner, log=lambda *a: None)
    assert c.mute() and c.unmute()
    assert any("get-source-mute" in cmd for cmd in calls)
    assert calls[-1] == linux_set_mute_cmd(False)     # was not muted, so restore unmutes


def test_windows_hotkey_sendkeys_zoom():
    cmd = windows_hotkey_sendkeys("zoom")
    assert cmd[0] == "powershell" and "^+a" in cmd[-1]


def test_mute_failure_is_logged_not_raised():
    logs = []
    def runner(cmd, timeout=3):
        raise RuntimeError("osascript blocked")
    c = MicController(method="volume", platform="darwin", runner=runner, log=lambda m: logs.append(m))
    assert c.mute() is False and not c.muted
    assert logs and "failed" in logs[0]


def test_close_restores_if_muted():
    calls = []
    def runner(cmd, timeout=3):
        calls.append(cmd)
        if "input volume of" in " ".join(cmd):
            return _Proc(0, "50\n")
        return _Proc(0, "")
    c = MicController(method="volume", platform="darwin", runner=runner, log=lambda *a: None)
    c.mute(); c.close()
    assert any("set volume input volume 50" in " ".join(cmd) for cmd in calls)
    assert not c.muted


def test_muted_badge_is_drawn_in_the_preview_corner():
    img = np.full((180, 320, 3), 30, np.uint8)
    out = draw_muted_badge(img.copy())
    # badge is top-right; it should be brighter / redder than the fill
    corner = out[10:50, 240:]
    assert corner[..., 2].mean() > 40          # some red (BGR)
    assert not np.array_equal(out, img)


# ---------- Effect: forward / reverse determinism and a visible dust cloud ----------
def _person(h=180, w=320):
    frame = np.full((h, w, 3), 40, np.float32)
    frame[:] = (180, 200, 215)                                   # wall
    frame[int(h * 0.72):] = (60, 90, 120)                        # desk
    mask = np.zeros((h, w), np.float32)
    # a standing person blob
    cv2.ellipse(mask, (w // 2, int(h * 0.62)), (40, 70), 0, 0, 360, 1.0, -1)
    cv2.circle(mask, (w // 2, int(h * 0.28)), 22, 1.0, -1)
    frame[mask > 0.5] = (70, 120, 200)                           # skin/shirt
    plate = frame.copy()
    plate[mask > 0.5] = (180, 200, 215)
    plate[int(h * 0.72):] = (60, 90, 120)
    # desk under the person in the plate too
    return frame, mask, plate


def test_snap_effect_is_deterministic_and_reversible():
    frame, mask, plate = _person()
    fwd = Effect("snap", mask, frame.shape, seed=7, frame=frame, n_particles=2000)
    a = fwd.render(frame, mask, plate, 0.7)
    b = fwd.render(frame, mask, plate, 0.7)
    assert np.allclose(a, b)
    # Same normalized t on the reverse effect matches the forward frame.
    rev = Effect("snap", mask, frame.shape, seed=7, frame=frame, n_particles=2000, reverse=True)
    t = 0.4
    fa = fwd.render(frame, mask, plate, t * fwd.duration)
    rb = rev.render(frame, mask, plate, (1.0 - t) * rev.duration)
    assert np.allclose(fa, rb, atol=1e-3)
    # Mid-dissolve is neither the intact person nor the empty plate, and dust left the body.
    mid = fwd.render(frame, mask, plate, 1.0)                    # 1 s into a 2 s dissolve
    assert float(np.abs(mid[mask > 0.5] - frame[mask > 0.5]).mean()) > 8
    assert float(np.abs(mid - plate).mean()) > 3
    # Some coloured flakes should have drifted outside the original silhouette.
    outside = (cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)
    delta = np.abs(mid - plate).mean(axis=2)
    assert int((delta[outside] > 20).sum()) > 40
    # When finished, the person is gone (empty room).
    end = fwd.render(frame, mask, plate, fwd.duration)
    assert float(np.abs(end - plate).mean()) < 1.5
    # Reverse at t=0 (elapsed=0, reversed -> t=1) is the empty room; at the end the person is back.
    r0 = rev.render(frame, mask, plate, 0.0)
    assert float(np.abs(r0 - plate).mean()) < 1.5
    r1 = rev.render(frame, mask, plate, rev.duration)
    assert float(np.abs(r1[mask > 0.5] - frame[mask > 0.5]).mean()) < 8


def test_snap_duration_is_two_seconds():
    assert DURATION["snap"] == 2.0
    ev = Effect("snap", np.ones((40, 40), np.float32), (40, 40, 3), n_particles=200)
    assert ev.duration == 2.0
    evr = Effect("snap", np.ones((40, 40), np.float32), (40, 40, 3), n_particles=200, reverse=True)
    assert abs(evr.duration - 2.0 * 0.85) < 1e-9


# ---------- Config / CLI ----------
def test_snap_gate_falls_back_when_mic_does_not_open(capsys):
    assert snap_gate_for({"snap_trigger": "audio"}, mic_ok=False).mode == "gesture"
    assert "falling back" in capsys.readouterr().out
    assert snap_gate_for({"snap_trigger": "both"}, mic_ok=True).mode == "both"
    assert snap_gate_for({"snap_trigger": "either"}, mic_ok=False).mode == "gesture"
    assert snap_gate_for({"snap_trigger": "gesture"}, mic_ok=False).mode == "gesture"


def test_parse_snap_flags_and_defaults():
    cfg = parse(["--no-sound", "--no-preview", "--snap-trigger", "both", "--no-auto-mute"])
    assert cfg["snap_trigger"] == "both"
    assert cfg["snap_auto_mute"] is False
    assert cfg["snap_reveal"] == "room"
    assert cfg["snap_particles"] == 6000
    assert "snap" in cfg["enabled_effects"]
    assert cfg["hold"]["snap"] == 0.0
    assert cfg["snap_pinch_frac"] == DEFAULTS["snap_pinch_frac"]
    assert cfg["snap_min_pinch"] == DEFAULTS["snap_min_pinch"]
    assert cfg["snap_flick_window"] == DEFAULTS["snap_flick_window"]
    assert cfg["snap_one_hand_only"] is True
    bad = normalize_snap({"snap_trigger": "nope", "snap_particles": "abc", "snap_reveal": "banana",
                          "snap_min_pinch": "nope", "snap_pinch_frac": 9})
    assert bad["snap_trigger"] == "gesture" and bad["snap_reveal"] == "room" and bad["snap_particles"] == 6000
    assert bad["snap_min_pinch"] == DEFAULTS["snap_min_pinch"]
    assert bad["snap_pinch_frac"] == 0.35                                  # clamped to the high end


# ---------- End-to-end on the synthetic clip (diff segmenter, no MediaPipe needed) ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "demo", "clips")


@pytest.mark.skipif(not os.path.exists(os.path.join(CLIPS, "person.mp4")), reason="demo clip missing")
def test_snap_vanish_and_reappear_end_to_end(tmp_path):
    out, log = str(tmp_path / "o.mp4"), str(tmp_path / "e.json")
    subprocess.run(
        [sys.executable, "meetingport.py", "--source", os.path.join(CLIPS, "person.mp4"),
         "--plate-seconds", "1", "--force", "snap@2.0,palm@4.4", "--no-preview", "--no-sound",
         "--segmenter", "diff", "--no-auto-mute", "--out", out, "--log-json", log],
        cwd=ROOT, check=True, capture_output=True)
    ev = json.load(open(log))
    assert [e["gesture"] for e in ev] == ["snap", "palm"]
    cap = cv2.VideoCapture(out); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f.astype(np.float32))
    cap0 = cv2.VideoCapture(os.path.join(CLIPS, "person.mp4"))
    room = np.median([cap0.read()[1].astype(np.float32) for _ in range(20)], axis=0)
    at = lambda t: fr[int(round((t - 1.0) * 30))]
    assert float(np.abs(at(1.8) - room).mean()) > 8          # person visible
    # Just after the snap, the body is still mostly there (the sweep takes ~2 s).
    # This also guards against firing with an empty mask (which would pop to the plate).
    early = at(2.15)
    assert float(np.abs(early - room).mean()) > 0.5 * float(np.abs(at(1.8) - room).mean())
    # mid-dissolve: not an empty room yet (dust still in the air / body half gone)
    mid = at(2.8)
    assert float(np.abs(mid - room).mean()) > 2
    assert float(np.abs(at(4.15) - room).mean()) < 3         # gone: empty room (default reveal)
    assert float(np.abs(at(6.5) - room).mean()) > 8          # back after reverse
