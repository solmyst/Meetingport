"""Gesture classifier + trigger state machine on synthetic landmarks (and real MediaPipe landmarks if available)."""
import os, sys, math, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mport import gestures as G

def hand(ext=("index", "middle", "ring", "pinky"), thumb=True, together=False, crossed=False, spread=False,
         cx=0.5, cy=0.6, s=0.1, rot=0.0):
    """Upright synthetic right hand: wrist below, fingers pointing up. s = wrist->middle MCP length."""
    pts = [None] * 21; pts[0] = (0, 0)
    bx = {"index": -0.35, "middle": -0.1, "ring": 0.12, "pinky": 0.32}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for f, i in base.items():
        x = bx[f]; pts[i] = (x, -1.0)
        if f in ext:
            tx = x
            if together and f in ("index", "middle"): tx = -0.22 + (0.02 if f == "middle" else -0.02)
            if spread and f in ("index", "middle"): tx = -0.85 if f == "index" else 0.35
            if crossed and f in ("index", "middle"): tx = -0.1 if f == "index" else -0.35
            pts[i + 1] = (x + (tx - x) * .4, -1.4); pts[i + 2] = (x + (tx - x) * .7, -1.65); pts[i + 3] = (tx, -1.9)
        else:
            pts[i + 1] = (x, -1.3); pts[i + 2] = (x, -1.05); pts[i + 3] = (x, -0.8)
    if thumb: pts[1:5] = [(-0.3, -0.2), (-0.6, -0.4), (-0.85, -0.6), (-1.05, -0.8)]
    else: pts[1:5] = [(-0.3, -0.2), (-0.4, -0.5), (-0.2, -0.7), (0.0, -0.75)]
    c, sn = math.cos(rot), math.sin(rot)
    return [(cx + s * (x * c - y * sn), cy + s * (x * sn + y * c)) for x, y in pts]

FACE = ((0.5, 0.25), 0.2)   # forehead point, face width

def goku_hand(**kw):  # fingertips placed on the forehead
    h = hand(ext=("index", "middle"), thumb=False, together=True, **kw)
    tip = ((h[8][0] + h[12][0]) / 2, (h[8][1] + h[12][1]) / 2)
    dx, dy = FACE[0][0] - tip[0], FACE[0][1] + 0.01 - tip[1]
    return [(x + dx, y + dy) for x, y in h]

def test_goku_detected():
    assert G.classify([goku_hand()], *FACE)[0] == G.GOKU

@pytest.mark.parametrize("rot", [-0.4, 0.4])
def test_goku_tilted(rot): assert G.classify([goku_hand(rot=rot)], *FACE)[0] == G.GOKU

def test_goku_needs_face(): assert G.classify([goku_hand()], None, None)[0] != G.GOKU

def test_two_fingers_away_from_forehead_not_goku():
    h = hand(ext=("index", "middle"), thumb=False, together=True, cx=0.2, cy=0.9)
    assert G.classify([h], *FACE)[0] != G.GOKU

def test_victory_spread_not_goku():
    h = hand(ext=("index", "middle"), thumb=False, spread=True)  # V sign, gap ~1.2 hand sizes like the real photo
    tip = h[8]; h = [(x + FACE[0][0] - tip[0], y + FACE[0][1] - tip[1]) for x, y in h]
    assert G.classify([h], *FACE)[0] != G.GOKU

def test_open_palm():
    assert G.classify([hand()], *FACE)[0] == G.PALM

def test_fist_nothing():
    assert G.classify([hand(ext=(), thumb=False)], *FACE)[0] is None

def test_pointing_nothing():
    assert G.classify([hand(ext=("index",), thumb=False)], *FACE)[0] is None

def test_gojo_crossed():
    assert G.classify([hand(ext=("index", "middle"), thumb=False, crossed=True, cy=0.8)], *FACE)[0] == G.GOJO

def test_naruto_two_hands_together():
    a = hand(ext=("index", "middle"), thumb=False, cx=0.48, cy=0.75)
    b = [(1.0 - x, y) for x, y in hand(ext=("index", "middle"), thumb=False, cx=0.48, cy=0.75)]  # mirrored left hand
    b = [(x - 0.04 + 0.04, y) for x, y in b]
    assert G.classify([a, b], *FACE)[0] == G.NARUTO

def test_naruto_flat_palms_pressed_at_chest():
    """The classic seal: both open-ish palms pressed together near the chest."""
    a = hand(cx=0.47, cy=0.72, s=0.11)
    b = [(1.0 - x, y) for x, y in hand(cx=0.47, cy=0.72, s=0.11)]
    # Pull them together so the tips touch.
    shift = 0.02
    b = [(x - shift, y) for x, y in b]
    assert G.classify([a, b], *FACE)[0] == G.NARUTO

def test_naruto_two_fists_pressed_is_not_a_seal():
    a = hand(ext=(), thumb=False, cx=0.48, cy=0.75)
    b = [(1.0 - x, y) for x, y in hand(ext=(), thumb=False, cx=0.48, cy=0.75)]
    assert G.classify([a, b], *FACE)[0] != G.NARUTO

def test_two_hands_apart_not_naruto():
    a = hand(cx=0.2); b = [(1 - x, y) for x, y in hand(cx=0.2)]
    assert G.classify([a, b], *FACE)[0] == G.PALM   # two open palms far apart = reappear, not seal

def test_naruto_beats_a_one_hand_snap_pose_on_the_same_frame():
    """While both hands form the seal, classify returns naruto (snap is handled separately and blocked)."""
    a = hand(ext=("index", "middle", "ring", "pinky"), cx=0.47, cy=0.72)
    b = [(1.0 - x + 0.02, y) for x, y in hand(ext=("index", "middle", "ring", "pinky"), cx=0.47, cy=0.72)]
    assert G.classify([a, b], *FACE)[0] == G.NARUTO

# ---------- trigger machine ----------
def run(seq, fps=30, cfg=None):
    m = G.TriggerMachine(cfg); fired = []
    for i, g in enumerate(seq):
        r = m.update(g, i / fps, 1 / fps)
        if r: fired.append((round(i / fps, 2), r))
    return fired, m

def test_hold_required():
    assert run([G.GOKU] * 15)[0] == []                       # 0.5s < 0.7s
    f, m = run([G.GOKU] * 25); assert f and f[0][1] == G.GOKU and not m.present

def test_brief_dropouts_tolerated():
    seq = ([G.GOKU] * 5 + [None] * 2) * 6                    # flaky detection, still mostly held
    assert run(seq)[0][0][1] == G.GOKU

def test_long_dropout_resets():
    seq = [G.GOKU] * 15 + [None] * 30 + [G.GOKU] * 10
    assert run(seq)[0] == []

def test_switching_gesture_restarts_charge():
    assert run([G.GOKU] * 12 + [G.GOJO] * 12)[0] == []

def test_palm_ignored_while_present_and_same_sign_brings_you_back():
    assert run([G.PALM] * 60)[0] == []
    f, m = run([G.GOKU] * 25 + [None] * 70 + [G.GOKU] * 40)
    assert [g for _, g in f] == [G.GOKU, G.PALM]

def test_cooldown_then_reappear():
    f, _ = run([G.GOKU] * 25 + [G.PALM] * 40)     # palm right away: blocked by 2s cooldown
    assert [g for _, g in f] == [G.GOKU]
    f, _ = run([G.GOKU] * 25 + [None] * 60 + [G.PALM] * 35)
    assert [g for _, g in f] == [G.GOKU, G.PALM]

def test_single_frame_noise_never_fires():
    import random; random.seed(0)
    seq = [random.choice([None] * 8 + [G.GOKU, G.NARUTO, G.GOJO, G.PALM]) for _ in range(3000)]
    assert run(seq)[0] == []

# ---------- real MediaPipe landmarks from Apache-2.0 MediaPipe sample photos ----------
ASSETS = os.path.join(os.path.dirname(__file__), "assets")
@pytest.mark.parametrize("img,expect_ok", [
    ("right_hands.jpg", lambda g: g == G.PALM), ("left_hands.jpg", lambda g: g == G.PALM),
    ("fist.jpg", lambda g: g is None), ("pointing_up.jpg", lambda g: g is None),
    ("thumbs_up.jpg", lambda g: g is None), ("victory.jpg", lambda g: g in (None, G.GOJO) and g != G.GOKU),
])
def test_real_mediapipe_landmarks(img, expect_ok):
    cv2 = pytest.importorskip("cv2"); pytest.importorskip("mediapipe")
    from mport.perception import HandTracker, FaceTracker
    f = cv2.imread(os.path.join(ASSETS, img)); assert f is not None
    hands = HandTracker()(f, 0); fh, fw = FaceTracker()(f, 0)
    g, _ = G.classify(hands, fh, fw)
    print(img, len(hands), g)
    assert hands and expect_ok(g)

# ---------- two-hand signs: Shadow Clone + Kamehameha ----------
def _move(h, idx, target):
    """Translate hand h so the mean of landmarks idx lands on target."""
    mx = sum(h[i][0] for i in idx) / len(idx); my = sum(h[i][1] for i in idx) / len(idx)
    return [(x + target[0] - mx, y + target[1] - my) for x, y in h]

AXIS = (5, 9, 8, 12)   # index/middle knuckles + tips

def clone_hands(angle=math.pi / 2, offset=(0.0, 0.0), rot_a=0.0):
    a = hand(ext=("index", "middle"), thumb=False, together=True, cx=0.5, cy=0.7, rot=rot_a)
    b = hand(ext=("index", "middle"), thumb=False, together=True, rot=rot_a + angle)
    c = (sum(a[i][0] for i in AXIS) / 4 + offset[0], sum(a[i][1] for i in AXIS) / 4 + offset[1])
    return [a, _move(b, AXIS, c)]

@pytest.mark.parametrize("angle", [math.pi / 2, -math.pi / 2, math.pi / 4, -math.pi / 4, 3 * math.pi / 4])
def test_clone_plus_and_x(angle):
    assert G.classify(clone_hands(angle), *FACE)[0] == G.CLONE

def test_clone_tilted_and_hand_order():
    hs = clone_hands(math.pi / 2, rot_a=0.3)
    assert G.classify(hs, *FACE)[0] == G.CLONE and G.classify(hs[::-1], *FACE)[0] == G.CLONE

def test_clone_wins_over_single_hand_gojo():
    a = hand(ext=("index", "middle"), thumb=False, crossed=True, cx=0.5, cy=0.7)   # crossed fingers would be gojo
    b = hand(ext=("index", "middle"), thumb=False, together=True, rot=math.pi / 2)
    c = (sum(a[i][0] for i in AXIS) / 4, sum(a[i][1] for i in AXIS) / 4)
    assert G.is_gojo(a) and G.classify([a, _move(b, AXIS, c)], *FACE)[0] == G.CLONE

def test_parallel_two_finger_hands_are_naruto_not_clone():
    a = hand(ext=("index", "middle"), thumb=False, cx=0.48, cy=0.75)
    b = [(1.0 - x, y) for x, y in hand(ext=("index", "middle"), thumb=False, cx=0.48, cy=0.75)]
    assert not G.is_clone([a, b]) and G.classify([a, b], *FACE)[0] == G.NARUTO

def test_clone_needs_fingers_to_touch():
    assert G.classify(clone_hands(math.pi / 2, offset=(0.15, 0.0)), *FACE)[0] != G.CLONE

def test_clone_needs_two_finger_shape():
    a, b = clone_hands(math.pi / 2)
    full = hand(rot=math.pi / 2); full = _move(full, AXIS, (sum(b[i][0] for i in AXIS) / 4, sum(b[i][1] for i in AXIS) / 4))
    assert G.classify([a, full], *FACE)[0] != G.CLONE

def test_single_gojo_still_gojo():
    assert G.classify([hand(ext=("index", "middle"), thumb=False, crossed=True, cy=0.8)], *FACE)[0] == G.GOJO

def kame_hands(open_angle=0.5, gap=0.0, spread=False):
    """Wrists together, the two hands opening outward like a mouth (right hand tilted right, left mirrored)."""
    a = hand(cx=0.5 + gap, cy=0.7, rot=open_angle, spread=spread)
    b = [(1.0 - x, y) for x, y in hand(cx=0.5 + gap, cy=0.7, rot=open_angle, spread=spread)]
    return [a, b]

@pytest.mark.parametrize("ang", [0.4, 0.7, 1.1])
def test_kamehameha_cupped_open(ang):
    assert G.classify(kame_hands(ang), *FACE)[0] == G.KAME

def test_kamehameha_curled_fingers_ok():
    a = hand(ext=("index", "middle", "ring"), cx=0.5, cy=0.7, rot=0.6)
    b = [(1.0 - x, y) for x, y in a]
    assert G.classify([a, b], *FACE)[0] == G.KAME

def test_flat_palms_pressed_together_are_naruto_not_kamehameha():
    a = hand(cx=0.48, cy=0.75); b = [(1.0 - x, y) for x, y in hand(cx=0.48, cy=0.75)]
    assert not G.is_kamehameha([a, b]) and G.classify([a, b], *FACE)[0] == G.NARUTO

def test_wrists_apart_is_palm_not_kamehameha():
    assert G.classify(kame_hands(0.5, gap=0.25), *FACE)[0] == G.PALM

def test_two_fists_wrists_together_not_kamehameha():
    a = hand(ext=(), thumb=False, cx=0.5, cy=0.7, rot=0.5); b = [(1.0 - x, y) for x, y in a]
    assert not G.is_kamehameha([a, b]) and G.classify([a, b], *FACE)[0] != G.KAME

# ---------- state: cloned + one-shot kamehameha ----------
def test_clone_enters_cloned_state_and_same_sign_dismisses():
    f, m = run([G.CLONE] * 25)
    assert [g for _, g in f] == [G.CLONE] and m.state == "cloned" and m.present
    f, m = run([G.CLONE] * 25 + [None] * 70 + [G.CLONE] * 25)
    assert [g for _, g in f] == [G.CLONE, G.UNCLONE] and m.state == "present"

def test_palm_dismisses_clones():
    f, m = run([G.CLONE] * 25 + [None] * 70 + [G.PALM] * 35)
    assert [g for _, g in f] == [G.CLONE, G.UNCLONE] and m.state == "present"

def test_vanish_and_kame_blocked_while_cloned():
    f, m = run([G.CLONE] * 25 + [None] * 70 + [G.GOKU] * 40 + [G.KAME] * 40)
    assert [g for _, g in f] == [G.CLONE] and m.state == "cloned"

def test_kamehameha_one_shot_stays_present():
    f, m = run([G.KAME] * 25)
    assert [g for _, g in f] == [G.KAME] and m.state == "present" and m.present
    f, m = run([G.KAME] * 25 + [None] * 70 + [G.KAME] * 25)
    assert [g for _, g in f] == [G.KAME, G.KAME] and m.state == "present"

def test_kamehameha_hold_and_not_while_gone():
    assert run([G.KAME] * 15)[0] == []                       # 0.5s < 0.7s
    f, m = run([G.GOKU] * 25 + [None] * 70 + [G.KAME] * 40)
    assert [g for _, g in f] == [G.GOKU] and m.state == "gone"

def test_vanish_and_reappear_unchanged_after_clones():
    f, m = run([G.CLONE] * 25 + [None] * 70 + [G.PALM] * 35 + [None] * 70 + [G.NARUTO] * 25 + [None] * 70 + [G.PALM] * 35)
    assert [g for _, g in f] == [G.CLONE, G.UNCLONE, G.NARUTO, G.PALM] and m.state == "present"

def test_old_config_without_new_hold_keys():
    m = G.TriggerMachine(G.TriggerConfig(hold={G.GOKU: 0.7, G.NARUTO: 0.6, G.GOJO: 0.8, G.PALM: 1.0}))
    fired = [m.update(G.CLONE, i / 30, 1 / 30) for i in range(25)]
    assert G.CLONE in fired

# ---------- state: Infinite Void (gojo no longer vanishes) ----------
def test_gojo_opens_void_and_does_not_vanish():
    f, m = run([G.GOJO] * 25)
    assert [g for _, g in f] == [G.GOJO] and m.state == "void" and m.present and m.voided and not m.cloned

def test_gojo_toggles_void_off():
    f, m = run([G.GOJO] * 25 + [None] * 70 + [G.GOJO] * 25)
    assert [g for _, g in f] == [G.GOJO, G.UNVOID] and m.state == "present" and not m.voided and m.present

def test_palm_dismisses_void_stays_present():
    f, m = run([G.GOJO] * 25 + [None] * 70 + [G.PALM] * 35)
    assert [g for _, g in f] == [G.GOJO, G.UNVOID] and m.state == "present" and m.present

def test_gojo_does_not_reappear_when_gone():
    f, m = run([G.GOKU] * 25 + [None] * 70 + [G.GOJO] * 40)
    assert [g for _, g in f] == [G.GOKU] and m.state == "gone" and not m.voided

def test_vanish_blocked_while_void():
    f, m = run([G.GOJO] * 25 + [None] * 70 + [G.GOKU] * 40 + [G.NARUTO] * 40)
    assert [g for _, g in f] == [G.GOJO] and m.state == "void" and m.present

def test_kamehameha_works_while_void():
    f, m = run([G.GOJO] * 25 + [None] * 70 + [G.KAME] * 25)
    assert [g for _, g in f] == [G.GOJO, G.KAME] and m.state == "void" and m.present

def test_clone_works_while_void():
    f, m = run([G.GOJO] * 25 + [None] * 70 + [G.CLONE] * 25)
    assert [g for _, g in f] == [G.GOJO, G.CLONE] and m.state == "void_cloned" and m.present and m.cloned and m.voided
    f, m = run([G.GOJO] * 25 + [None] * 70 + [G.CLONE] * 25 + [None] * 70 + [G.CLONE] * 25)
    assert [g for _, g in f] == [G.GOJO, G.CLONE, G.UNCLONE] and m.state == "void" and m.voided and not m.cloned

def test_gojo_works_while_cloned():
    f, m = run([G.CLONE] * 25 + [None] * 70 + [G.GOJO] * 25)
    assert [g for _, g in f] == [G.CLONE, G.GOJO] and m.state == "void_cloned" and m.cloned and m.voided
    # palm dismisses the void first; clones remain
    f, m = run([G.CLONE] * 25 + [None] * 70 + [G.GOJO] * 25 + [None] * 70 + [G.PALM] * 35)
    assert [g for _, g in f] == [G.CLONE, G.GOJO, G.UNVOID] and m.state == "cloned" and m.cloned and not m.voided

def test_palm_while_present_still_ignored_without_void():
    assert run([G.PALM] * 60)[0] == []
