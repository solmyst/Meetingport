"""Meetingport v2 main loop."""
import argparse, json, os, sys, time, cv2, numpy as np
from . import gestures as G
from .effects import Effect, draw_charge_ring, render_clones
from .perception import MPSegmenter, DiffSegmenter, HandTracker, FaceTracker, PlateBuilder, PlateCapture
from .sounds import ensure_sounds, play
from .mute import MicController, draw_muted_badge
from .access import ensure_access

DEFAULTS = {
    "source": "0", "width": 1280, "height": 720, "fps": 30,
    "segmenter": "mediapipe",          # mediapipe | diff (v1 fallback)
    "plate_seconds": 4.0,              # empty-room capture at start (extended until clean); 0 = skip, learn automatically
    "recapture_countdown": 3.0,
    "plate_learn": False,              # keep refreshing the plate after the capture (opt-in: can learn ghosts)        # B key: seconds to step out before the clean recapture starts
    "gestures": True, "enabled_effects": ["goku", "naruto", "gojo", "clone", "kamehameha", "snap"],
    "hold": {"goku": 0.7, "naruto": 0.6, "gojo": 0.8, "palm": 1.0, "clone": 0.7, "kamehameha": 0.7, "snap": 0.0}, "cooldown": 2.0,
    "snap_trigger": "gesture",         # gesture | audio | both | either
    "snap_reveal": "room",             # room | log  (log uses the same Kawarimi overlay as Naruto)
    "snap_particles": 6000,
    "snap_auto_mute": True,
    "auto_unmute_on_reappear": True,
    # Stricter snap detector (defaults reject talking-hand false positives):
    "snap_pinch_frac": 0.18,           # thumb-middle distance, in hand sizes, to count as a pinch
    "snap_min_pinch": 0.10,            # seconds the snap-ready pinch must be held before a flick counts
    "snap_flick_window": 0.28,         # max seconds from pinch release to a completed flick
    "snap_min_sep": 0.38,              # thumb-middle separation after the flick, in hand sizes
    "snap_min_travel": 0.30,           # middle-tip travel toward the palm, in hand sizes
    "snap_max_wrist_move": 0.12,       # wrist motion allowed during the gesture, in hand sizes
    "snap_gesture_cooldown": 0.85,     # refractory after a completed snap gesture
    "snap_one_hand_only": True,        # never arm/fire a snap while two hands are visible (Naruto/clone/kame)
    "mute_method": "volume",           # volume (system input) | hotkey (meeting app shortcut)
    "meeting_app": "meet",             # meet | zoom | teams  (used only by mute_method=hotkey)
    "gojo_void_seconds": 0,            # 0 = domain stays until dismissed; else auto-collapse after this many seconds fully open
    "gojo_void_caption": True,         # on-screen "Domain Expansion: Infinite Void"
    "naruto_substitution": "log",      # log | none | image  (Kawarimi log after a Naruto vanish)
    "substitution_image": "",          # PNG with alpha, used when naruto_substitution is "image"
    "substitution_scale": 1.0,         # size multiplier for the log / custom image
    "substitution_drop_seconds": 0.55, # how long the Kawarimi log takes to fall in (with the smoke)
    "substitution_drop_height": 1.15,  # drop start height as a multiple of the log height
    "sound": True, "virtualcam": False, "preview": True,
    "ring_in_output": False,           # charging ring is on the preview only unless true
    "mirror": False, "out": None, "force": "", "log_json": None, "max_seconds": None,
}

def parse(argv=None):
    ap = argparse.ArgumentParser(description="Meetingport v2: anime-gesture teleport out of video calls")
    ap.add_argument("--config", help="JSON config file (CLI flags override it)")
    ap.add_argument("--source", help="camera index (0) or a video file")
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int); ap.add_argument("--fps", type=int)
    ap.add_argument("--segmenter", choices=["mediapipe", "diff"])
    ap.add_argument("--plate-seconds", type=float, dest="plate_seconds")
    ap.add_argument("--plate-learn", dest="plate_learn", action="store_const", const=True,
                    help="keep learning the background after the capture (off by default)")
    ap.add_argument("--no-gestures", dest="gestures", action="store_const", const=False)
    ap.add_argument("--no-sound", dest="sound", action="store_const", const=False)
    ap.add_argument("--virtualcam", action="store_const", const=True)
    ap.add_argument("--no-preview", dest="preview", action="store_const", const=False)
    ap.add_argument("--ring-in-output", dest="ring_in_output", action="store_const", const=True)
    ap.add_argument("--mirror", action="store_const", const=True)
    ap.add_argument("--out", help="also record the output to this .mp4")
    ap.add_argument("--substitution", choices=["log", "none", "image"], dest="naruto_substitution",
                    help="after a Naruto vanish: leave a wooden log (default), a custom image, or an empty room")
    ap.add_argument("--snap-trigger", choices=["gesture", "audio", "both", "either"], dest="snap_trigger",
                    help="how the Thanos snap fires: gesture, audio click, both within 300 ms, or either")
    ap.add_argument("--no-auto-mute", dest="snap_auto_mute", action="store_const", const=False,
                    help="do not mute the mic when the snap vanish fires")
    ap.add_argument("--force", help='scripted triggers for testing, e.g. "goku@2,palm@4.5"')
    ap.add_argument("--log-json", dest="log_json", help="write trigger events to this JSON file")
    ap.add_argument("--max-seconds", type=float, dest="max_seconds")
    a = vars(ap.parse_args(argv)); cfg = dict(DEFAULTS)
    if a.get("config"):
        user = json.load(open(a["config"])); cfg.update(user)
        cfg["hold"] = {**DEFAULTS["hold"], **user.get("hold", {})}   # older configs lack the new hold keys
    cfg.update({k: v for k, v in a.items() if v is not None and k != "config"})
    return normalize_snap(normalize_substitution(cfg))

def _float_cfg(cfg, key, default, lo=None, hi=None):
    try:
        v = float(cfg[key]) if cfg.get(key) not in (None, "") else float(default)
    except (TypeError, ValueError):
        v = float(default)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def normalize_substitution(cfg):
    """Fill / coerce Kawarimi substitution keys so older configs and CLI still work."""
    mode = str(cfg.get("naruto_substitution") or "log").lower().strip()
    if mode not in ("log", "none", "image"):
        mode = "log"
    cfg["naruto_substitution"] = mode
    raw = cfg.get("substitution_scale")
    try:
        scale = float(1.0 if raw in (None, "") else raw)
    except (TypeError, ValueError):
        scale = 1.0
    cfg["substitution_scale"] = 1.0 if scale <= 0 else scale
    cfg["substitution_image"] = cfg.get("substitution_image") or ""
    cfg["substitution_drop_seconds"] = _float_cfg(
        cfg, "substitution_drop_seconds", DEFAULTS["substitution_drop_seconds"], 0.0, 3.0)
    cfg["substitution_drop_height"] = _float_cfg(
        cfg, "substitution_drop_height", DEFAULTS["substitution_drop_height"], 0.0, 4.0)
    return cfg

def normalize_snap(cfg):
    """Coerce snap-to-dust keys so older configs and CLI still work."""
    trig = str(cfg.get("snap_trigger") or "gesture").lower().strip()
    if trig not in ("gesture", "audio", "both", "either"):
        trig = "gesture"
    cfg["snap_trigger"] = trig
    reveal = str(cfg.get("snap_reveal") or "room").lower().strip()
    if reveal not in ("room", "log"):
        reveal = "room"
    cfg["snap_reveal"] = reveal
    try:
        n = int(cfg.get("snap_particles") if cfg.get("snap_particles") not in (None, "") else 6000)
    except (TypeError, ValueError):
        n = 6000
    cfg["snap_particles"] = int(max(200, min(20000, n)))
    cfg["snap_auto_mute"] = bool(cfg.get("snap_auto_mute", True))
    cfg["auto_unmute_on_reappear"] = bool(cfg.get("auto_unmute_on_reappear", True))
    cfg["snap_pinch_frac"] = _float_cfg(cfg, "snap_pinch_frac", DEFAULTS["snap_pinch_frac"], 0.08, 0.35)
    cfg["snap_min_pinch"] = _float_cfg(cfg, "snap_min_pinch", DEFAULTS["snap_min_pinch"], 0.0, 1.0)
    cfg["snap_flick_window"] = _float_cfg(cfg, "snap_flick_window", DEFAULTS["snap_flick_window"], 0.08, 1.0)
    cfg["snap_min_sep"] = _float_cfg(cfg, "snap_min_sep", DEFAULTS["snap_min_sep"], 0.15, 1.0)
    cfg["snap_min_travel"] = _float_cfg(cfg, "snap_min_travel", DEFAULTS["snap_min_travel"], 0.10, 1.0)
    cfg["snap_max_wrist_move"] = _float_cfg(cfg, "snap_max_wrist_move", DEFAULTS["snap_max_wrist_move"], 0.02, 1.0)
    cfg["snap_gesture_cooldown"] = _float_cfg(cfg, "snap_gesture_cooldown", DEFAULTS["snap_gesture_cooldown"], 0.0, 5.0)
    cfg["snap_one_hand_only"] = bool(cfg.get("snap_one_hand_only", True))
    method = str(cfg.get("mute_method") or "volume").lower().strip()
    cfg["mute_method"] = method if method in ("volume", "hotkey") else "volume"
    cfg["meeting_app"] = str(cfg.get("meeting_app") or "meet").strip().lower() or "meet"
    return cfg

def snap_watcher_for(cfg):
    """Build the strict SnapWatcher from config (and the built-in defaults)."""
    return G.SnapWatcher(
        pinch_frac=cfg.get("snap_pinch_frac", DEFAULTS["snap_pinch_frac"]),
        min_pinch=cfg.get("snap_min_pinch", DEFAULTS["snap_min_pinch"]),
        flick_window=cfg.get("snap_flick_window", DEFAULTS["snap_flick_window"]),
        min_sep=cfg.get("snap_min_sep", DEFAULTS["snap_min_sep"]),
        min_travel=cfg.get("snap_min_travel", DEFAULTS["snap_min_travel"]),
        max_wrist_move=cfg.get("snap_max_wrist_move", DEFAULTS["snap_max_wrist_move"]),
        cooldown=cfg.get("snap_gesture_cooldown", DEFAULTS["snap_gesture_cooldown"]),
        one_hand_only=cfg.get("snap_one_hand_only", True),
    )

def snap_gate_for(cfg, mic_ok):
    """Audio modes need an open microphone. If it failed to open, use the finger-snap gesture."""
    mode = cfg.get("snap_trigger", "gesture")
    if mode in ("audio", "both", "either") and not mic_ok:
        print("snap trigger falling back to gesture (no mic)")
        mode = "gesture"
    return G.SnapGate(mode=mode, window=0.30)

def open_source(cfg):
    s = str(cfg["source"]); is_file = not s.isdigit()
    cap = cv2.VideoCapture(s if is_file else int(s))
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"]); cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    if not cap.isOpened(): sys.exit(f"cannot open source {s} (on macOS grant Camera permission to your terminal)")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["fps"]
    return cap, is_file, (fps if fps and fps < 121 else cfg["fps"])

class Meetingport:
    def __init__(self, cfg, shape, fps):
        self.cfg, self.shape, self.fps = cfg, shape, fps
        self.seg = MPSegmenter() if cfg["segmenter"] == "mediapipe" else DiffSegmenter()
        self.hands = HandTracker() if cfg["gestures"] else None
        self.face = FaceTracker() if cfg["gestures"] else None
        self.plate = PlateBuilder(shape)
        hold = {**G.TriggerConfig().hold, **dict(cfg["hold"])}
        self.trig = G.TriggerMachine(G.TriggerConfig(hold=hold, cooldown=cfg["cooldown"]))
        self.sfx = ensure_sounds() if cfg["sound"] else {}
        self.effect, self.effect_t0, self.last_kind = None, 0, "goku"
        self.events, self.anchor, self.charging_kind = [], None, None
        self.clones = None                   # (layout, bbox) while the Shadow Clones are out
        self.capture, self.cap_start = None, 0.0
        self.void_fx, self.void_vis = None, False
        self.void_origin, self.void_t0, self.void_hold_t0 = None, 0.0, 0.0
        self.substitution, self.sub_bbox, self.sub_t0 = None, None, 0.0   # Kawarimi log while vanished
        self.snap_watch = snap_watcher_for(cfg)
        self.mic = None
        mic_ok = False
        if cfg.get("gestures") and cfg.get("snap_trigger") in ("audio", "both", "either"):
            from .audio import MicSnapListener
            self.mic = MicSnapListener()
            mic_ok = bool(self.mic.start())
            if not mic_ok:
                self.mic = None
        self.snap_gate = snap_gate_for(cfg, mic_ok)
        self.muter = MicController(method=cfg.get("mute_method", "volume"),
                                   meeting_app=cfg.get("meeting_app", "meet"))

    # ---------- clean-plate capture (startup and B key) ----------
    def start_capture(self, now, countdown=0.0):
        self.capture = PlateCapture(self.shape, max(1, int(max(self.cfg["plate_seconds"], 1.0) * self.fps)))
        self.cap_start = now + countdown
        if countdown > 0: print(f"Recapturing background in {countdown:.0f}s: step OUT of the camera view")

    def _capture_step(self, frame, now, mask):
        if now < self.cap_start: return
        self.capture.add(frame, mask)
        if self.capture.done():
            c = self.capture; self.plate = PlateBuilder(self.shape); self.plate.seed_capture(c); self.capture = None
            miss = 1 - float(c.result()[1].mean())
            print(f"Background captured from {c.frames} frames ({c.clean_fraction() * 100:.1f}% clean"
                  + (f", {miss * 100:.1f}% never visible, filled in and learned later" if miss > 0 else "") + "). Come back in!")

    def capture_hud(self, img, now):
        """Countdown / progress + red warning in the preview while capturing."""
        c = self.capture
        if c is None: return img
        h, w = img.shape[:2]
        if now < self.cap_start: msg, col = f"Step OUT of frame: capturing background in {self.cap_start - now:.1f}s", (0, 220, 255)
        else:
            left = max(0.0, (c.target - c.frames) / self.fps)
            msg = ("PAUSED: step out of frame" if c.person_in_frame() else
                   f"Capturing empty room... {left:.1f}s" if left > 0 else "Capturing empty room... waiting for clean pixels") \
                  + f"  ({c.clean_fraction() * 100:.0f}% clean)"; col = (255, 255, 255)
        cv2.rectangle(img, (0, 0), (w, 44), (0, 0, 0), -1); cv2.putText(img, msg, (12, 30), 0, 0.8, col, 2, cv2.LINE_AA)
        if c.person_in_frame():
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.rectangle(img, (0, h - 50), (w, h), (0, 0, 160), -1)
            cv2.putText(img, "You're still in frame! Step out of view", (12, h - 16), 0, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def _ensure_void(self):
        if self.void_fx is None:
            from .void import get_void
            self.void_fx = get_void(self.shape)
        return self.void_fx

    def fire(self, kind, now, mask, src, frame=None):
        """Manual/forced trigger. Vanish only while present, palm (reappear) only while gone.
        Gojo toggles the Infinite Void and never vanishes you. Palm dismisses an open domain first."""
        if self.effect is not None: return
        if not self.trig.present:
            if kind == "gojo": return                    # gojo is not a reappear / vanish sign
            kind = "palm"                                # same vanish sign (or palm) brings you back
        elif kind == "gojo":
            if kind not in self.cfg["enabled_effects"] and not self.trig.voided: return
            if self.trig.voided: kind = "unvoid"
        elif self.trig.voided and kind == "palm":
            kind = "unvoid"
        elif self.trig.cloned:                           # clone sign again (or palm) dismisses the clones
            if kind not in ("clone", "palm"): return
            kind = "unclone"
        elif kind == "palm" or kind not in self.cfg["enabled_effects"]: return
        self.trig.fire(kind, now); self._start(kind, now, mask, src, frame=frame)

    def _fx_kwargs(self, kind, frame=None):
        if kind != "snap":
            return {}
        return dict(frame=frame, n_particles=self.cfg.get("snap_particles", 6000),
                    anchor=self.anchor)

    def _start(self, kind, now, mask, src, frame=None):
        if kind == "palm":
            self.effect = Effect(self.last_kind, mask, self.shape, reverse=True,
                                 **self._fx_kwargs(self.last_kind, frame))
            snd = self.sfx.get((self.last_kind, True)); what = "reappear"
            self._maybe_unmute()
        elif kind == "clone":
            self.effect = Effect("clone", mask, self.shape); self.clones = (self.effect.layout, self.effect.bbox)
            snd = self.sfx.get(("clone", False)); what = "shadow clones out"
        elif kind == "unclone":
            lay, bb = self.clones or (None, None)
            self.effect = Effect("clone", mask, self.shape, reverse=True, layout=lay)
            if bb is not None: self.effect.bbox = bb
            snd = self.sfx.get(("clone", True)); what = "clones dismissed"
        elif kind == "kamehameha":
            self.effect = Effect("kamehameha", mask, self.shape, anchor=self.anchor if src == "gesture" else None)
            snd = self.sfx.get(("kamehameha", False)); what = "one-shot (you stay)"
        elif kind == "gojo":
            self.void_origin = self.anchor
            self.void_t0 = now; self.void_vis = True
            self.effect = Effect("gojo", mask, self.shape, anchor=self.anchor, void=self._ensure_void())
            snd = self.sfx.get(("gojo", False)); what = "domain expansion: infinite void"
        elif kind == "unvoid":
            self.effect = Effect("gojo", mask, self.shape, reverse=True,
                                 anchor=self.anchor or self.void_origin, void=self._ensure_void())
            snd = None; what = "domain collapse"
        else:
            self.last_kind = kind; self.effect = Effect(kind, mask, self.shape, **self._fx_kwargs(kind, frame))
            snd = self.sfx.get((kind, False)); what = "vanish"
            self._arm_substitution(kind, mask, now=now)
            if kind == "snap":
                self._maybe_mute()
        self.effect_kind = kind
        self.effect_t0 = now; self.events.append({"t": round(now, 3), "gesture": kind, "source": src})
        print(f"[{now:6.2f}s] {src}: {kind} -> {what}")
        if snd: play(snd)

    def _arm_substitution(self, kind, mask, now=None):
        """Remember the person bbox and build the Kawarimi overlay for a Naruto vanish."""
        from .substitution import prepare_substitution, person_bbox
        use_log = kind == "naruto" or (kind == "snap" and self.cfg.get("snap_reveal") == "log")
        if not use_log:
            self.substitution, self.sub_bbox, self.sub_t0 = None, None, 0.0
            return
        # Snap-to-log reuses the Kawarimi overlay; mode comes from snap_reveal or naruto_substitution.
        mode = self.cfg.get("naruto_substitution", "log") if kind == "naruto" else "log"
        if kind == "snap":
            mode = "log"
        self.sub_bbox = person_bbox(mask)
        self.sub_t0 = float(now if now is not None else 0.0)
        self.substitution = prepare_substitution(
            mode, mask, self.shape,
            image_path=self.cfg.get("substitution_image") or "",
            scale=self.cfg.get("substitution_scale", 1.0))

    def cycle_substitution(self):
        """Preview-window 'L' key: cycle log / empty room (and a custom image if configured)."""
        from .substitution import prepare_substitution
        modes = ["log", "none"]
        if self.cfg.get("substitution_image"):
            modes.append("image")
        cur = self.cfg.get("naruto_substitution") or "log"
        nxt = modes[0] if cur not in modes else modes[(modes.index(cur) + 1) % len(modes)]
        self.cfg["naruto_substitution"] = nxt
        mid_naruto = self.effect is not None and self.effect.kind == "naruto"
        if (not self.trig.present or mid_naruto) and self.sub_bbox is not None:
            if nxt == "none":
                self.substitution = None
            else:
                x0, y0, x1, y1 = self.sub_bbox
                mask = np.zeros(self.shape[:2], np.float32)
                mask[y0:y1 + 1, x0:x1 + 1] = 1
                self.substitution = prepare_substitution(
                    nxt, mask, self.shape,
                    image_path=self.cfg.get("substitution_image") or "",
                    scale=self.cfg.get("substitution_scale", 1.0))
        print(f"Naruto substitution: {nxt}")
        return nxt

    def _plate_with_substitution(self, plate, now):
        """Composite the log onto the clean plate: falls in with the smoke, then sits until reappear."""
        if self.substitution is None:
            return plate
        from .substitution import composite_substitution, drop_y_offset
        from .effects import naruto_visibility
        alpha = 1.0
        if self.effect is not None and self.effect.kind == "naruto":
            t = float(np.clip((now - self.effect_t0) / max(1e-6, self.effect.duration), 0, 1))
            if self.effect.reverse:
                t = 1.0 - t
            alpha = 1.0 - naruto_visibility(t)
        elif self.effect is not None and self.effect.kind == "snap":
            t = float(np.clip((now - self.effect_t0) / max(1e-6, self.effect.duration), 0, 1))
            if self.effect.reverse:
                t = 1.0 - t
            alpha = float(np.clip((t - 0.25) / 0.50, 0, 1))
        elif self.trig.present:
            return plate
        rgba, placement = self.substitution
        # During reappear the log is already seated; only animate the forward drop.
        reversing = self.effect is not None and self.effect.reverse
        drop = 0.0
        if not reversing:
            drop_s = self.cfg.get("substitution_drop_seconds", 0.55)
            drop_h = placement["height"] * self.cfg.get("substitution_drop_height", 1.15)
            drop = drop_y_offset(now - self.sub_t0, drop_s, drop_h)
        return composite_substitution(plate, rgba, placement, alpha=alpha, drop=drop)

    def _live_bbox(self, mask):
        """Person bbox from a 1/8 mask (cheap), so the clones follow you as you move; fire-time bbox if empty."""
        sm = cv2.resize(mask, (mask.shape[1] // 8, mask.shape[0] // 8), interpolation=cv2.INTER_AREA)
        ys, xs = np.where(sm > 0.5)
        if len(xs) < 20: return self.clones[1]
        return (xs.min() * 8, ys.min() * 8, xs.max() * 8 + 8, ys.max() * 8 + 8)

    def _live_mic(self):
        """Only touch the real microphone for a live camera, never for a file source (tests/demos)."""
        return str(self.cfg.get("source", "0")).isdigit()

    def _maybe_mute(self):
        if self.cfg.get("snap_auto_mute", True) and self._live_mic():
            ok = self.muter.mute()
            if not ok:
                print("auto-mute: could not mute the mic (see log); continuing without it")

    def _maybe_unmute(self):
        if self.cfg.get("auto_unmute_on_reappear", True) and self.last_kind == "snap" and self._live_mic():
            self.muter.unmute()

    def close(self):
        try:
            self.muter.close()
        except Exception:
            pass
        if self.mic is not None:
            try: self.mic.close()
            except Exception: pass
        for o in (getattr(self.seg, "seg", None), getattr(self.hands, "det", None), getattr(self.face, "det", None)):
            try: o and o.close()
            except Exception: pass

    def step(self, frame, now, dt):
        ts = int(now * 1000)
        mask = self.seg(frame, ts, self.plate.plate if self.plate.known.min() > 0 else None)
        if self.capture is not None:                       # (re)capturing the room: pass the camera through
            self._capture_step(frame, now, mask)
            return frame.copy() if self.trig.present else np.clip(self.plate.get(), 0, 255).astype(np.uint8)
        self.n = getattr(self, "n", 0) + 1
        if self.cfg.get("plate_learn") and self.trig.present and self.effect is None and self.n % 5 == 0:
            safe = cv2.dilate((mask > 0.02).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(np.float32)
            self.plate.update(frame, safe, rate=0.2)  # never learn pixels near you into the background
        plate = self._plate_with_substitution(self.plate.get(), now)
        if self.hands is not None:
            hands = self.hands(frame, ts); fh, fw = self.face(frame, ts)
            g, anchor = G.classify(hands, fh, fw)
            if g is not None and g != "palm" and g not in self.cfg["enabled_effects"]: g = None
            snap_g, snap_anc = self.snap_watch.update(hands, now)
            audio_snap = bool(self.mic and self.mic.poll(now))
            # Named signs and two-hand seals win: never let a snap steal a Naruto/clone/kame charge.
            two_hand_block = bool(self.cfg.get("snap_one_hand_only", True)) and len(hands) >= 2
            named_blocks = g is not None or two_hand_block
            snap_ok = self.snap_gate.update(bool(snap_g) and not named_blocks,
                                            audio_snap and not two_hand_block, now)
            if snap_ok and "snap" in self.cfg["enabled_effects"]:
                g, anchor = G.SNAP, snap_anc or anchor
            H = frame.shape[0]
            self.anchor = (anchor[0] * H, anchor[1] * H) if anchor else None
            if self.effect is None:
                fired = self.trig.update(g, now, dt)
                if fired: self._start(fired, now, mask, "gesture", frame=frame)
            self.charging_kind = self.trig.charging
        # Auto-collapse the domain after gojo_void_seconds of being fully open (0 = until dismissed).
        hold_s = self.cfg.get("gojo_void_seconds", 0) or 0
        if self.trig.voided and self.effect is None and hold_s > 0 and now - self.void_hold_t0 >= hold_s:
            self.trig.fire("unvoid", now); self._start("unvoid", now, mask, "timeout")

        f = frame.astype(np.float32)
        # While the domain is fully open (or some other effect is playing over it), put the person on the void first.
        void_hold = self.void_vis and (self.effect is None or getattr(self, "effect_kind", None) not in ("gojo", "unvoid"))
        if void_hold:
            cap = bool(self.cfg.get("gojo_void_caption", True))
            f = self._ensure_void().render(f, mask, coverage=1.0, clock=now - self.void_t0,
                                           origin=self.void_origin, caption=cap)
        if self.effect is not None:
            if self.effect.kind == "gojo":
                self.effect.anim_time = now - self.void_t0
                self.effect.caption = bool(self.cfg.get("gojo_void_caption", True))
            out = self.effect.render(f, mask, plate, now - self.effect_t0)
            # Keep clones visible during the domain open/close animation.
            if self.effect.kind == "gojo" and self.clones is not None:
                out = render_clones(out, mask, self.clones[0], self._live_bbox(mask))
            if self.effect.done(now - self.effect_t0):
                if self.effect_kind == "unclone": self.clones = None
                if self.effect_kind == "gojo": self.void_hold_t0 = now
                if self.effect_kind == "unvoid": self.void_vis = False
                if self.effect_kind == "palm": self.substitution = None
                self.effect = None
        elif self.trig.present and self.clones is not None:
            out = render_clones(f, mask, self.clones[0], self._live_bbox(mask))   # clones persist, live, every frame
        elif self.trig.present or self.void_vis: out = f
        else: out = plate
        return np.clip(out, 0, 255).astype(np.uint8)


def main(argv=None):
    cfg = parse(argv)
    ensure_access(cfg)
    cap, is_file, fps = open_source(cfg)
    ok, frame = cap.read()
    if not ok: sys.exit("no frames from source")
    if cfg["mirror"]: frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]; mp_ = Meetingport(cfg, frame.shape, fps)
    forced = sorted([(float(x.split("@")[1]), x.split("@")[0].strip()) for x in cfg["force"].split(",") if "@" in x])

    # 1) clean plate: step out of frame. Every frame is segmented and each pixel is only learned from frames where
    #    you're not near it; the capture runs past plate_seconds until enough clean frames exist (or 3x the time).
    t_start = time.monotonic(); n_cap = 1; pending = frame
    if cfg["plate_seconds"] > 0:
        print(f"Step OUT of the camera view: capturing the empty room for {cfg['plate_seconds']:.0f}s ...")
        mp_.start_capture(0.0)
        while mp_.capture is not None:
            if pending is None:
                ok, pending = cap.read()
                if not ok: break
                if cfg["mirror"]: pending = cv2.flip(pending, 1)
                n_cap += 1
            now = (n_cap - 1) / fps if is_file else time.monotonic() - t_start
            mp_.step(pending, now, 1 / fps)
            if cfg["preview"] and not is_file:
                pv = mp_.capture_hud(pending.copy(), now); cv2.imshow("Meetingport preview", pv); cv2.waitKey(1)
            pending = None
        if mp_.capture is not None:                 # source ended early: use what we have
            mp_.plate.seed_capture(mp_.capture); mp_.capture = None
    frames = [None] * n_cap

    cam = None
    if cfg["virtualcam"]:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(w, h, fps, fmt=pyvirtualcam.PixelFormat.BGR); print("Virtual camera:", cam.device)
    vw = cv2.VideoWriter(cfg["out"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if cfg["out"] else None
    n, last, t0 = 0, 0.0, t_start; hud = True   # one clock from startup (segmenter timestamps must keep increasing)
    print("Keys: G=Goku  N=Naruto  S=Snap to dust  J=Gojo Infinite Void (again or R to dismiss)  C=Shadow Clones (again to dismiss)  K=Kamehameha  V=vanish  R=reappear / dismiss void  L=cycle Naruto log/empty  B=recapture background (3s countdown)  H=hide ring  Q=quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        if cfg["mirror"]: frame = cv2.flip(frame, 1)
        n += 1; now = (n + len(frames) - 1) / fps if is_file else time.monotonic() - t0
        if cfg["max_seconds"] and now > cfg["max_seconds"]: break
        while forced and now >= forced[0][0]:
            _, k = forced.pop(0); mp_.fire(k, now, getattr(mp_.seg, "prev", None) if getattr(mp_.seg, "prev", None) is not None else np.zeros((h, w), np.float32), "forced", frame=frame)
        out = mp_.step(frame, now, now - last); last = now
        prog = mp_.trig.progress()
        if cfg["ring_in_output"]: draw_charge_ring(out, mp_.anchor, prog, mp_.charging_kind)
        if cam: cam.send(out); cam.sleep_until_next_frame()
        if vw: vw.write(out)
        if cfg["preview"]:
            pv = out.copy()
            if hud: draw_charge_ring(pv, mp_.anchor, prog, mp_.charging_kind)
            if mp_.muter.muted: draw_muted_badge(pv)
            mp_.capture_hud(pv, now)
            cv2.imshow("Meetingport preview", pv); k = cv2.waitKey(1) & 0xFF
            keys = {ord('g'): "goku", ord('n'): "naruto", ord('s'): "snap", ord('j'): "gojo", ord('c'): "clone", ord('k'): "kamehameha", ord('v'): mp_.last_kind, ord('r'): "palm"}
            if k in keys:
                kind = keys[k]
                mp_.fire(kind, now, getattr(mp_.seg, "prev", None) if getattr(mp_.seg, "prev", None) is not None else np.zeros((h, w), np.float32), "key", frame=frame)
            elif k == ord('b') and mp_.capture is None: mp_.start_capture(now, cfg["recapture_countdown"])
            elif k == ord('l'): mp_.cycle_substitution()
            elif k == ord('h'): hud = not hud
            elif k in (ord('q'), 27): break
    if vw: vw.release()
    if cam: cam.close()
    cap.release(); cv2.destroyAllWindows(); mp_.close()
    if cfg["log_json"]: json.dump(mp_.events, open(cfg["log_json"], "w"), indent=1)
    print(f"done: {n} frames, events={mp_.events}")
    return mp_.events

if __name__ == "__main__": main()
