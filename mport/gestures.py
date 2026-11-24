"""Pure-Python gesture classification on MediaPipe hand landmarks + a debounced trigger state machine.
Landmarks: list of 21 (x, y) in normalized image coords (0..1, y down). No MediaPipe import here -> unit-testable."""
from dataclasses import dataclass, field
import math

WRIST, THUMB_TIP = 0, 4
MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

GOKU, NARUTO, GOJO, PALM = "goku", "naruto", "gojo", "palm"
CLONE, KAME, UNCLONE, UNVOID = "clone", "kamehameha", "unclone", "unvoid"
SNAP = "snap"
VANISH_GESTURES = (GOKU, NARUTO, SNAP)                    # gojo is a domain toggle, not a vanish
PRESENT_GESTURES = VANISH_GESTURES + (CLONE, KAME, GOJO)   # what can charge while you're visible

def _d(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
def hand_size(h): return max(_d(h[WRIST], h[MCP["middle"]]), 1e-6)

def extended(h, f):
    """Finger straight: tip clearly farther from wrist than PIP, and roughly collinear."""
    return _d(h[TIP[f]], h[WRIST]) > 1.15 * _d(h[PIP[f]], h[WRIST]) and \
           _d(h[TIP[f]], h[MCP[f]]) > 0.55 * hand_size(h)

def curled(h, f): return not extended(h, f)

def thumb_out(h):
    return _d(h[THUMB_TIP], h[MCP["pinky"]]) > 0.9 * hand_size(h)

def is_open_palm(h):
    if not all(extended(h, f) for f in MCP) or not thumb_out(h): return False
    s = hand_size(h)  # fingers spread a bit
    return _d(h[TIP["index"]], h[TIP["pinky"]]) > 0.6 * s

def two_fingers(h):
    return extended(h, "index") and extended(h, "middle") and curled(h, "ring") and curled(h, "pinky")

def fingers_together(h):
    return _d(h[TIP["index"]], h[TIP["middle"]]) < 0.45 * hand_size(h)

def fingers_crossed(h):
    """Middle finger crosses over index: the left/right order of tips is flipped vs the knuckles."""
    a = h[MCP["index"]][0] - h[MCP["middle"]][0]
    b = h[TIP["index"]][0] - h[TIP["middle"]][0]
    return a * b < 0 and abs(b) > 0.06 * hand_size(h)

def is_goku(h, forehead, face_w):
    """Index+middle straight, pressed together, crossed-not, tips on the forehead."""
    if forehead is None or not two_fingers(h) or not fingers_together(h) or fingers_crossed(h): return False
    tip = ((h[TIP["index"]][0] + h[TIP["middle"]][0]) / 2, (h[TIP["index"]][1] + h[TIP["middle"]][1]) / 2)
    return _d(tip, forehead) < 0.55 * face_w

def is_gojo(h, forehead=None, face_w=None):
    """Crossed index/middle (Gojo's 'Domain Expansion' sign), not at the forehead."""
    if not (extended(h, "index") and extended(h, "middle") and curled(h, "ring") and curled(h, "pinky")): return False
    return fingers_crossed(h)

def is_naruto(hands):
    """Two hands pressed together (hand seal): wrists and fingertips of both hands close, near-ish chest height."""
    if len(hands) < 2: return False
    a, b = hands[0], hands[1]
    s = (hand_size(a) + hand_size(b)) / 2
    tips_close = min(_d(a[TIP[f]], b[TIP[g]]) for f in TIP for g in TIP) < 0.45 * s
    wrists_close = _d(a[WRIST], b[WRIST]) < 1.3 * s
    centers = _d(_center(a), _center(b)) < 1.0 * s
    return tips_close and wrists_close and centers

def _center(h): return (sum(p[0] for p in h) / 21, sum(p[1] for p in h) / 21)
def _mid(a, b): return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def _finger_axis(h):
    """Segment from the index/middle knuckles to the index/middle tips (the 'blade' of a two-finger sign)."""
    return _mid(h[MCP["index"]], h[MCP["middle"]]), _mid(h[TIP["index"]], h[TIP["middle"]])

def _seg_dist(p, q, r, s):
    """Min distance between segments pq and rs (0 if they intersect)."""
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    if cross(p, q, r) * cross(p, q, s) < 0 and cross(r, s, p) * cross(r, s, q) < 0: return 0.0
    def pt(a, b, c):
        vx, vy = b[0] - a[0], b[1] - a[1]; L = vx * vx + vy * vy or 1e-12
        u = max(0.0, min(1.0, ((c[0] - a[0]) * vx + (c[1] - a[1]) * vy) / L))
        return _d(c, (a[0] + u * vx, a[1] + u * vy))
    return min(pt(p, q, r), pt(p, q, s), pt(r, s, p), pt(r, s, q))

def is_clone(hands):
    """Shadow Clone seal: both hands show index+middle straight (ring/pinky curled), and one hand's two fingers
    cross the other's at a steep angle (a plus or an X). Parallel two-finger hands pressed together = naruto."""
    if len(hands) < 2: return False
    a, b = hands[0], hands[1]
    if not (two_fingers(a) and two_fingers(b)): return False
    s = (hand_size(a) + hand_size(b)) / 2
    (a0, a1), (b0, b1) = _finger_axis(a), _finger_axis(b)
    va, vb = (a1[0] - a0[0], a1[1] - a0[1]), (b1[0] - b0[0], b1[1] - b0[1])
    cos = abs(va[0] * vb[0] + va[1] * vb[1]) / (math.hypot(*va) * math.hypot(*vb) + 1e-9)
    if cos > math.cos(math.radians(40)): return False        # need >= 40 deg between the finger pairs
    return _seg_dist(a0, a1, b0, b1) < 0.35 * s                # and the fingers actually overlap

def _open_hand(h):
    """Fingers open (>= 3 of the 4 straight) and not squeezed together."""
    n = sum(extended(h, f) for f in MCP)
    return n >= 3 and _d(h[TIP["index"]], h[TIP["pinky"]]) > 0.45 * hand_size(h)

def is_kamehameha(hands):
    """Kamehameha cup: both wrists close together, both hands open/spread, fingertips of the two hands apart
    (a 'mouth' opening between the palms) - unlike naruto, where the flat hands touch tip to tip."""
    if len(hands) < 2: return False
    a, b = hands[0], hands[1]
    if not (_open_hand(a) and _open_hand(b)): return False
    s = (hand_size(a) + hand_size(b)) / 2
    wrists_close = _d(a[WRIST], b[WRIST]) < 0.9 * s
    tips_apart = min(_d(a[TIP[f]], b[TIP[g]]) for f in TIP for g in TIP) > 0.5 * s
    return wrists_close and tips_apart

def classify(hands, forehead=None, face_w=None):
    """Return (gesture or None, anchor point for the charging ring). Two-hand signs are checked first."""
    if is_clone(hands): return CLONE, _mid(_center(hands[0]), _center(hands[1]))
    if is_kamehameha(hands): return KAME, _mid(_center(hands[0]), _center(hands[1]))
    if is_naruto(hands): return NARUTO, _center(hands[0])
    for h in hands:
        if is_goku(h, forehead, face_w): return GOKU, h[TIP["index"]]
    for h in hands:
        if is_gojo(h): return GOJO, h[TIP["index"]]
    for h in hands:
        if is_open_palm(h): return PALM, _center(h)
    return None, None

def is_snap_pinch(h, frac=0.25):
    """Thumb tip and middle fingertip pressed together (the start of a snap)."""
    return _d(h[THUMB_TIP], h[TIP["middle"]]) < frac * hand_size(h)


class SnapWatcher:
    """Pinch, then a quick middle-finger flick toward the palm, within flick_window seconds.

    This is a motion, not a held pose: update() returns (True, anchor) only on the
    completing frame. Pure Python, no MediaPipe, unit-testable with synthetic landmarks.
    """
    def __init__(self, pinch_frac=0.25, flick_window=0.40, min_sep=0.30,
                 min_travel=0.22, min_dt=0.033):
        self.pinch_frac = pinch_frac
        self.flick_window = flick_window
        self.min_sep = min_sep          # thumb-middle distance after the flick, in hand sizes
        self.min_travel = min_travel    # middle tip travel toward the palm, in hand sizes
        self.min_dt = min_dt
        self._tracks = []

    def reset(self):
        self._tracks = []

    def update(self, hands, now):
        """Return (fired, anchor_xy or None). fires at most once, then clears tracks."""
        fired, anchor = False, None
        hands = list(hands or [])
        used = set()
        for tr in self._tracks:
            j = self._match(tr, hands, used)
            if j is None:
                continue
            used.add(j)
            ev, anc = self._step(tr, hands[j], now)
            if ev:
                fired, anchor = True, anc
        for j, h in enumerate(hands):
            if j in used:
                continue
            if is_snap_pinch(h, self.pinch_frac):
                self._tracks.append(self._new(h, now))
        self._tracks = [tr for tr in self._tracks if now - tr["last"] <= self.flick_window + 0.08]
        if fired:
            self._tracks = []
        return fired, anchor

    def _new(self, h, now):
        return {
            "pinch_t": now, "last": now,
            "middle": h[TIP["middle"]], "thumb": h[THUMB_TIP], "wrist": h[WRIST],
            "size": hand_size(h),
        }

    def _match(self, tr, hands, used):
        best, bd = None, 0.15
        for j, h in enumerate(hands):
            if j in used:
                continue
            d = _d(h[WRIST], tr["wrist"])
            if d < bd:
                best, bd = j, d
        return best

    def _step(self, tr, h, now):
        tr["last"] = now
        if is_snap_pinch(h, self.pinch_frac):
            tr["pinch_t"] = now
            tr["middle"] = h[TIP["middle"]]
            tr["thumb"] = h[THUMB_TIP]
            tr["wrist"] = h[WRIST]
            tr["size"] = hand_size(h)
            return False, None
        dt = now - tr["pinch_t"]
        if dt > self.flick_window or dt < self.min_dt:
            return False, None
        s = tr["size"]
        sep = _d(h[THUMB_TIP], h[TIP["middle"]]) / s
        mid = h[TIP["middle"]]
        toward = (tr["wrist"][0] - tr["middle"][0], tr["wrist"][1] - tr["middle"][1])
        move = (mid[0] - tr["middle"][0], mid[1] - tr["middle"][1])
        tlen = math.hypot(*toward) + 1e-9
        proj = (move[0] * toward[0] + move[1] * toward[1]) / tlen
        if sep >= self.min_sep and (proj / s) >= self.min_travel:
            return True, mid
        return False, None


class SnapGate:
    """Fuse a completed snap gesture and an optional audio click per snap_trigger mode.

    Modes: "gesture" | "audio" | "both" | "either". "both" requires the two events
    within `window` seconds (default 0.3). Each accepted event fires once.
    """
    def __init__(self, mode="gesture", window=0.30):
        mode = (mode or "gesture").lower().strip()
        if mode not in ("gesture", "audio", "both", "either"):
            mode = "gesture"
        self.mode, self.window = mode, float(window)
        self.g_t = self.a_t = self._fired_at = -1e9

    def update(self, gesture_snap, audio_snap, now):
        if gesture_snap:
            self.g_t = now
        if audio_snap:
            self.a_t = now
        if self.mode == "gesture":
            want = bool(gesture_snap)
        elif self.mode == "audio":
            want = bool(audio_snap)
        elif self.mode == "either":
            want = bool(gesture_snap or audio_snap)
        else:  # both
            if gesture_snap and (now - self.a_t) <= self.window:
                want = True
            elif audio_snap and (now - self.g_t) <= self.window:
                want = True
            else:
                want = False
        # Refractory = the fusion window, so a gesture+audio pair counts as one snap.
        if want and now - self._fired_at > self.window:
            self._fired_at = now
            return True
        return False


@dataclass
class TriggerConfig:
    hold: dict = field(default_factory=lambda: {GOKU: 0.7, NARUTO: 0.6, GOJO: 0.8, PALM: 1.0, CLONE: 0.7, KAME: 0.7, SNAP: 0.0})
    cooldown: float = 2.0      # seconds after any effect before another can charge
    grace: float = 0.25        # tolerate brief detection dropouts
    decay: float = 2.0         # charge drains this many times faster than it fills when lost

class TriggerMachine:
    """States: present -> (vanish sign held) -> gone -> (same sign or palm held) -> present.
    present -> (clone sign held) -> cloned -> (clone sign or palm held, fires "unclone") -> present.
    present -> (gojo sign held) -> void -> (gojo sign or palm held, fires "unvoid") -> present.
    Void can overlap with clones. Kamehameha still works while the void is open (one-shot, you stay).
    Vanish signs are ignored while the void is active. Charge tolerates short dropouts."""
    def __init__(self, cfg=None):
        self.cfg = cfg or TriggerConfig(); self.present = True; self.cloned = False; self.voided = False
        self.charging, self.charge, self.last_seen, self.last_fire = None, 0.0, -1e9, -1e9

    @property
    def state(self):
        if not self.present: return "gone"
        if self.voided: return "void_cloned" if self.cloned else "void"
        return "cloned" if self.cloned else "present"

    def hold(self, g): return self.cfg.hold.get(g, 0.7)

    def allowed(self, g):
        if not self.present: return g == PALM or g in VANISH_GESTURES   # gojo no longer brings you back
        if self.voided: return g in (GOJO, PALM, CLONE, KAME)           # vanish is gated; clone/kame still work
        if self.cloned: return g in (CLONE, PALM, GOJO)                 # domain can open while clones are out
        return g in PRESENT_GESTURES

    def resolve(self, g):
        """What a completed charge of g actually does in the current state."""
        if not self.present: return PALM                 # vanish sign or palm brings you back
        if g == GOJO: return UNVOID if self.voided else GOJO
        if self.voided and g == PALM: return UNVOID      # palm dismisses the domain first (clones stay)
        if self.cloned and g in (CLONE, PALM): return UNCLONE
        return g

    def update(self, gesture, now, dt):
        """Returns fired action name or None. self.progress() gives 0..1 for the ring."""
        if now - self.last_fire < self.cfg.cooldown:
            self.charging, self.charge = None, 0.0; return None
        if gesture is not None and self.allowed(gesture):
            if gesture != self.charging: self.charging, self.charge = gesture, 0.0
            self.charge += dt; self.last_seen = now
        elif self.charging is not None:
            if now - self.last_seen > self.cfg.grace:
                self.charge -= dt * self.cfg.decay
                if self.charge <= 0: self.charging, self.charge = None, 0.0
        if self.charging and self.charge >= self.hold(self.charging):
            g = self.resolve(self.charging); self.fire(g, now); return g
        return None

    def fire(self, g, now):
        if g == CLONE: self.cloned = True
        elif g == UNCLONE: self.cloned = False
        elif g == KAME: pass                             # one-shot, state unchanged
        elif g == GOJO: self.voided = True               # domain on; you stay visible
        elif g == UNVOID: self.voided = False
        else: self.present = (g == PALM); self.cloned = False; self.voided = False
        self.last_fire = now; self.charging, self.charge = None, 0.0

    def progress(self):
        return 0.0 if not self.charging else min(1.0, self.charge / self.hold(self.charging))
