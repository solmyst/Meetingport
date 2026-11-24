"""Anime-style effects. Vanish effects are pure functions of normalized time t (0..1) so they can be
played backwards for the reappear. Gojo is a toggled Infinite Void backdrop (person stays visible).
Frames are float32 BGR 0..255; mask is HxW float 0..1 (1 = person)."""
import cv2, numpy as np

DURATION = {"goku": 1.0, "naruto": 1.3, "gojo": 1.1, "clone": 1.3, "kamehameha": 1.9, "snap": 2.0}
RING_COLOR = {"goku": (255, 220, 120), "naruto": (60, 160, 255), "gojo": (230, 80, 200), "palm": (120, 255, 140),
              "clone": (40, 150, 255), "kamehameha": (255, 200, 90), "snap": (180, 80, 230)}

def _bbox(mask):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0: h, w = mask.shape; return w // 3, h // 4, 2 * w // 3, h
    return xs.min(), ys.min(), xs.max(), ys.max()

def _sample(mask, n, rng):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape; return np.c_[rng.uniform(w * .3, w * .7, n), rng.uniform(h * .3, h, n)]
    i = rng.integers(0, len(xs), n); return np.c_[xs[i], ys[i]].astype(np.float32) + rng.normal(0, 2, (n, 2))

def _smooth(a, b, t): return float(np.clip((t - a) / (b - a), 0, 1)) ** 2 * (3 - 2 * float(np.clip((t - a) / (b - a), 0, 1)))

def naruto_visibility(t):
    """Person opacity during the Naruto puff. t is 0..1, already un-reversed."""
    return 1.0 - _smooth(0.08, 0.16, t)

class Effect:
    """Created once when the trigger fires; samples particles from the person's silhouette at that moment."""
    def __init__(self, kind, mask, shape, reverse=False, seed=7, layout=None, anchor=None, void=None,
                 frame=None, n_particles=6000, sweep=None):
        self.kind, self.reverse = kind, reverse
        self.duration = DURATION[kind] * (0.85 if reverse and kind != "gojo" else 1.0)
        rng = np.random.default_rng(seed); self.rng = rng
        h, w = shape[:2]; self.h, self.w = h, w
        self.bbox = _bbox(mask); x0, y0, x1, y1 = self.bbox
        self.scale = max(8.0, (y1 - y0) / 6)
        if kind == "naruto":
            n = 160; self.p0 = _sample(mask, n, rng)
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            out = self.p0 - c; out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-6
            self.v = out * rng.uniform(0.6, 1.6, (n, 1)) * self.scale * 1.1 + np.array([0, -self.scale * 0.8])
            self.r0 = rng.uniform(0.5, 1.1, n) * self.scale * 0.9
            self.shade = rng.uniform(0.82, 1.0, n)
        elif kind == "gojo":
            # Infinite Void: expand from the hand (or mid-torso). The person stays visible.
            if anchor is None: anchor = ((x0 + x1) / 2, y0 + 0.58 * (y1 - y0))
            self.ax, self.ay = float(anchor[0]), float(anchor[1])
            from .void import get_void
            self.void = void or get_void(shape)
            self.caption = True
            self.anim_time = 0.0
        elif kind == "clone":
            self.layout = layout or clone_layout(self.bbox, w)
            n = 70; self.puffs = []
            for dx, sc in self.layout:              # person-shaped puff at every spawn point
                p = _sample(mask, n, rng); cx, by = (x0 + x1) / 2, y1
                p = np.c_[cx + (p[:, 0] - cx) * sc + dx, by + (p[:, 1] - by) * sc]
                ang = rng.uniform(0, 2 * np.pi, n)
                v = np.c_[np.cos(ang), np.sin(ang) * 0.6 - 0.5] * rng.uniform(0.3, 1.2, (n, 1)) * self.scale * 0.9
                self.puffs.append((p, v, rng.uniform(0.6, 1.2, n) * self.scale * 0.8 * sc, rng.uniform(0.8, 1.0, n)))
        elif kind == "kamehameha":
            if anchor is None: anchor = ((x0 + x1) / 2, y0 + 0.62 * (y1 - y0))   # hands at chest height
            self.ax, self.ay = float(anchor[0]), float(anchor[1])
            self.dir = 1 if self.ax <= w / 2 else -1                              # fire toward the far edge
            self.R = max(14.0, h * 0.075)
            n = 70; ang = rng.uniform(0, 2 * np.pi, n)                            # sparks spiralling into the ball
            self.sp_ang, self.sp_r = ang, rng.uniform(2.0, 5.0, n) * self.R
            self.sp_spd, self.sp_ph = rng.uniform(0.6, 1.4, n), rng.uniform(0, 1, n)
            self.shake = rng.normal(0, 1, (64, 2))
            self.streak_y = rng.uniform(-0.8, 0.8, 18); self.streak_o = rng.uniform(0, 1, 18)
        elif kind == "snap":
            self._init_snap(mask, frame, rng, n_particles, sweep, anchor)
        elif kind == "goku":
            self.lines_x = rng.uniform(x0 - 20, x1 + 20, 40); self.lines_len = rng.uniform(0.3, 1.0, 40)

    def _noise_field(self, mask, rng):
        h, w = mask.shape
        n = cv2.resize(rng.random((max(2, h // 24), max(2, w // 24))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        n = cv2.normalize(n, None, 0, 1, cv2.NORM_MINMAX)
        y0, y1 = self.bbox[1], self.bbox[3]
        ramp = np.clip((np.arange(h)[:, None] - y0) / max(1, y1 - y0), 0, 1)  # dissolve from top of head down
        return np.clip(0.55 * n + 0.45 * ramp, 0, 1).astype(np.float32)

    def done(self, elapsed): return elapsed >= self.duration

    def render(self, frame, mask, plate, elapsed):
        t = float(np.clip(elapsed / self.duration, 0, 1))
        if self.reverse: t = 1 - t
        return getattr(self, "_" + self.kind)(frame, mask, plate, t)

    # ---------- composite helper ----------
    @staticmethod
    def _comp(frame, plate, m, alpha):
        """Person over plate. The soft mask is re-feathered narrowly so its faint tail (dilation, blur, temporal
        smoothing) can't leave a translucent ghost, and alpha 0 returns the pure plate."""
        if np.isscalar(alpha) and alpha <= 0: return plate.copy()
        m = np.clip((m - 0.25) / 0.5, 0, 1)
        a = (m * alpha)[..., None]; return frame * a + plate * (1 - a)

    # ---------- Goku: Instant Transmission ----------
    def _goku(self, frame, mask, plate, t):
        k = int(t * 40)
        if t < 0.35: vis = 1.0
        elif t < 0.75: vis = (0.75 if k % 2 == 0 else 0.15) * (1 - (t - 0.35) / 0.4)
        else: vis = 0.0
        out = self._comp(frame, plate, mask, vis)
        if 0.35 <= t < 0.8:  # afterimage: blue-tinted ghost drifting up
            dy = int(-(t - 0.35) * self.scale * 2)
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            gm = cv2.warpAffine(mask, M, (self.w, self.h)) * 0.35 * (1 - (t - 0.35) / 0.45)
            gf = cv2.warpAffine(frame, M, (self.w, self.h)) * np.array([1.3, 1.0, 0.6], np.float32)
            out = out * (1 - gm[..., None]) + np.clip(gf, 0, 255) * gm[..., None]
        glow_amt = _smooth(0.0, 0.33, t) * (1 - _smooth(0.4, 0.8, t))
        if glow_amt > 0:  # aura around silhouette
            sm = cv2.resize(mask, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
            edge = cv2.resize(cv2.GaussianBlur(sm, (0, 0), self.scale * 0.125), (self.w, self.h)) - mask * 0.6
            edge = np.clip(edge, 0, 1)[..., None] * glow_amt
            out = out + edge * np.array([255, 210, 140], np.float32) * 1.6
            x0, y0, x1, y1 = self.bbox
            for x, L in zip(self.lines_x, self.lines_len):  # speed lines rising
                ya = int(y1 - (y1 - y0) * L * glow_amt * 1.4); cv2.line(out, (int(x), int(y1)), (int(x), ya), (255, 235, 200), 1, cv2.LINE_AA)
        flash = np.exp(-((t - 0.38) / 0.05) ** 2) * 0.9
        if flash > 0.01: out = out * (1 - flash) + np.array([255, 240, 215], np.float32) * flash
        return np.clip(out, 0, 255)

    # ---------- smoke helper (low-res soft puffs, shaded top-lit like anime smoke) ----------
    def _smoke(self, out, puffs, dens_list):
        s = 4; A = np.zeros((self.h // s + 1, self.w // s + 1), np.float32); S = np.zeros_like(A)
        for (pos, r, sh), dens in zip(puffs, dens_list):
            if dens <= 0: continue
            for (x, y), rr, c in zip(pos, r, sh):
                cv2.circle(A, (int(x / s), int(y / s)), max(1, int(rr / s)), float(dens), -1, cv2.LINE_AA)
                cv2.circle(S, (int(x / s), int(y / s - rr / s * 0.3)), max(1, int(rr / s * 0.7)), float(c * dens), -1, cv2.LINE_AA)
        if A.max() <= 0: return out
        A = cv2.GaussianBlur(A, (0, 0), 3); S = cv2.GaussianBlur(S, (0, 0), 4); S = S / (cv2.GaussianBlur(A, (0, 0), 4) + 1e-3)
        A = cv2.resize(A, (self.w, self.h))[: self.h, : self.w]; S = cv2.resize(S, (self.w, self.h))[: self.h, : self.w]
        A = np.clip(A, 0, 0.95)[..., None]
        smoke = (150 + 100 * np.clip(S, 0, 1))[..., None] * np.array([1.0, 1.0, 1.02], np.float32)
        return out * (1 - A) + smoke * A

    # ---------- Shadow Clone Jutsu: smoke poofs, then copies of you ----------
    def _clone(self, frame, mask, plate, t):
        n = len(self.layout); out = frame
        pops = [0.22 + 0.1 * i for i in range(n)]                      # staggered "poof!" per clone
        alphas, scales, dens = [], [], []
        for i, t0 in enumerate(pops):
            a = _smooth(t0 - 0.02, t0 + 0.04, t)
            bump = 1 + 0.12 * np.exp(-((t - t0 - 0.06) / 0.06) ** 2)      # tiny overshoot as it pops in
            alphas.append(a); scales.append(bump)
            tt = np.clip((t - t0 + 0.12) / 0.75, 0, 1)
            dens.append(0.0 if tt <= 0 else min(1.0, tt / 0.12) * (1 - _smooth(0.35, 1.0, tt)))
        out = render_clones(frame, mask, self.layout, self.bbox, alphas, scales)
        puffs = []
        for (p, v, r0, sh), t0 in zip(self.puffs, pops):
            tt = np.clip((t - t0 + 0.12) / 0.75, 0, 1); e = 1 - (1 - tt) ** 3
            puffs.append((p + v * e, r0 * (0.5 + 1.2 * e), sh))
        return np.clip(self._smoke(out, puffs, dens), 0, 255)

    # ---------- Kamehameha: charging ball, then a big beam ----------
    def _kamehameha(self, frame, mask, plate, t):
        T = self.duration * t                                    # seconds (one-shot: never reversed in the app)
        charge_end, beam_end = 0.42, 0.88
        h, w, R = self.h, self.w, self.R
        q = 4; Lh, Lw = h // q + 1, w // q + 1
        glow = np.zeros((Lh, Lw, 3), np.float32)                 # low-res emission, bloomed
        core = np.zeros((h // 2 + 1, w // 2 + 1, 3), np.float32) # half-res crisp core
        blue, white = np.array([255, 170, 60], np.float32), np.array([255, 250, 235], np.float32)
        # shake starts with the beam
        shake = 0.0 if t < charge_end else 1.0 * (1 - _smooth(charge_end, beam_end + 0.1, t))
        k = int(T * 30) % len(self.shake); sx, sy = (self.shake[k] * shake * h * 0.012).astype(int)
        # ball
        grow = _smooth(0.0, charge_end, t)
        if t > beam_end: grow *= 1 - _smooth(beam_end, 1.0, t)
        flick = 1 + 0.08 * np.sin(T * 60) + 0.05 * np.sin(T * 37)
        r = R * (0.15 + 0.85 * grow) * flick * (1.25 if charge_end <= t < beam_end else 1.0)
        ax, ay = self.ax, self.ay
        if grow > 0.01:
            cv2.circle(glow, (int(ax / q), int(ay / q)), max(1, int(r * 2.2 / q)), (blue * 0.55).tolist(), -1, cv2.LINE_AA)
            cv2.circle(glow, (int(ax / q), int(ay / q)), max(1, int(r * 1.3 / q)), (blue * 0.9).tolist(), -1, cv2.LINE_AA)
            cv2.circle(core, (int(ax / 2), int(ay / 2)), max(1, int(r / 2)), (blue * 0.9 + 25).tolist(), -1, cv2.LINE_AA)
            cv2.circle(core, (int(ax / 2), int(ay / 2)), max(1, int(r * 0.6 / 2)), white.tolist(), -1, cv2.LINE_AA)
        # sparks spiralling in while charging, spraying off the ball during the beam
        if t < beam_end:
            ph = (T * self.sp_spd * 1.6 + self.sp_ph) % 1.0
            rad = self.sp_r * (1 - ph) + r * ph
            ang = self.sp_ang + ph * 3.0
            vis = (ph > 0.05) & (np.random.default_rng(k).random(len(ph)) > 0.25)
            for a_, rr_, v_ in zip(ang, rad, vis):
                if not v_: continue
                x, y = ax + np.cos(a_) * rr_, ay + np.sin(a_) * rr_
                x2, y2 = ax + np.cos(a_ - 0.25) * (rr_ + R * 0.5), ay + np.sin(a_ - 0.25) * (rr_ + R * 0.5)
                cv2.line(core, (int(x / 2), int(y / 2)), (int(x2 / 2), int(y2 / 2)), (255, 235, 190), 1, cv2.LINE_AA)
                cv2.circle(glow, (int(x / q), int(y / q)), 1, (blue * 0.8).tolist(), -1)
        # beam
        if t >= charge_end:
            ext = _smooth(charge_end, charge_end + 0.08, t)
            thin = 1 - _smooth(beam_end - 0.05, 1.0, t)
            end_x = ax + self.dir * ext * (w + R * 4)
            xs = np.linspace(ax, end_x, 48)
            u = np.abs(xs - ax) / max(1.0, w)
            thick = R * (0.9 + 0.6 * np.minimum(1, u * 3)) * thin * (1 + 0.07 * np.sin(xs * 0.05 - T * 40))
            for scale_, col, layer, div in ((1.9, blue * 0.6, glow, q), (1.0, blue * 0.95 + 20, core, 2), (0.55, white, core, 2)):
                top = np.c_[xs, ay - thick * scale_]; bot = np.c_[xs[::-1], ay + thick[::-1] * scale_]
                poly = (np.r_[top, bot] / div).astype(np.int32)
                cv2.fillPoly(layer, [poly], col.tolist(), cv2.LINE_AA)
            for yo, o in zip(self.streak_y, self.streak_o):         # streaks racing along the beam
                L = (o + T * 2.5) % 1.0; x0 = ax + self.dir * L * w; x1 = x0 + self.dir * w * 0.12
                yy = ay + yo * R * thin
                cv2.line(core, (int(x0 / 2), int(yy / 2)), (int(x1 / 2), int(yy / 2)), (255, 255, 250), 1, cv2.LINE_AA)
        # composite: slight darkening for drama, shake the scene, add bloom + core
        dim = 0.22 * _smooth(0.0, 0.3, t) * (1 - _smooth(0.9, 1.0, t))
        out = frame * (1 - dim) + np.array([60, 20, 0], np.float32) * dim
        bloom = cv2.GaussianBlur(glow, (0, 0), max(1.0, R / q * 0.9)) * 1.6 + cv2.GaussianBlur(glow, (0, 0), max(1.0, R / q * 3)) * 1.2
        bloom = cv2.resize(bloom, (w, h))[:h, :w]
        c = cv2.resize(cv2.GaussianBlur(core, (0, 0), 1.2), (w, h))[:h, :w]
        out = out + bloom + c * 1.1
        if sx or sy: out = cv2.warpAffine(out, np.float32([[1, 0, sx], [0, 1, sy]]), (w, h), borderMode=cv2.BORDER_REFLECT)
        flash = 0.55 * np.exp(-((t - charge_end - 0.01) / 0.025) ** 2)
        if flash > 0.01: out = out * (1 - flash) + white * flash
        return np.clip(out, 0, 255)

    # ---------- Naruto: substitution smoke puff ----------
    def _naruto(self, frame, mask, plate, t):
        vis = naruto_visibility(t)
        out = self._comp(frame, plate, mask, vis)
        tt = np.clip((t - 0.05) / 0.95, 0, 1)
        if tt <= 0: return np.clip(out, 0, 255)
        ease = 1 - (1 - tt) ** 3
        pos = self.p0 + self.v * ease
        r = self.r0 * (0.5 + 1.3 * ease)
        dens = min(1.0, tt / 0.08) * (1 - _smooth(0.45, 1.0, tt))
        s = 4; A = np.zeros((self.h // s + 1, self.w // s + 1), np.float32); S = np.zeros_like(A)
        for (x, y), rr, sh in zip(pos, r, self.shade):
            cv2.circle(A, (int(x / s), int(y / s)), max(1, int(rr / s)), 1.0, -1, cv2.LINE_AA)
            cv2.circle(S, (int(x / s), int(y / s - rr / s * 0.3)), max(1, int(rr / s * 0.7)), float(sh), -1, cv2.LINE_AA)
        A = cv2.GaussianBlur(A, (0, 0), 3); S = cv2.GaussianBlur(S, (0, 0), 4); S = S / (cv2.GaussianBlur(A, (0, 0), 4) + 1e-3)
        A = cv2.resize(A, (self.w, self.h))[: self.h, : self.w]; S = cv2.resize(S, (self.w, self.h))[: self.h, : self.w]
        A = np.clip(A * dens, 0, 0.95)[..., None]
        smoke = (150 + 100 * np.clip(S, 0, 1))[..., None] * np.array([1.0, 1.0, 1.02], np.float32)
        return np.clip(out * (1 - A) + smoke * A, 0, 255)

    # ---------- Gojo: Domain Expansion, Infinite Void (person stays; background becomes the void) ----------
    def _gojo(self, frame, mask, plate, t):
        # t is openness 0..1 (already flipped when collapsing). clock keeps the starfield moving.
        clock = getattr(self, "anim_time", (1.0 - t) * self.duration if self.reverse else t * self.duration)
        return self.void.render(frame, mask, coverage=t, clock=clock,
                                origin=(self.ax, self.ay), caption=getattr(self, "caption", True))


    def _init_snap(self, mask, frame, rng, n_particles, sweep, anchor):
        """Sample flakes from the person at the trigger instant; precompute a closed-form path."""
        h, w = self.h, self.w
        n = int(n_particles) if n_particles is not None else 6000
        n = int(np.clip(n, 200, 20000))
        src = None if frame is None else np.asarray(frame, dtype=np.float32)
        if src is not None and src.ndim == 2:
            src = np.repeat(src[..., None], 3, axis=2)
        self.frozen = None if src is None else src.copy()
        self.mask0 = np.asarray(mask, dtype=np.float32)
        # Sweep from the snapping-hand side when we know it, otherwise from the right (the reel's look).
        if sweep in ("left", "right"):
            from_right = sweep == "right"
        elif anchor is not None:
            from_right = float(anchor[0]) >= (w / 2)
        else:
            from_right = True
        self.sweep_from_right = from_right
        xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        if from_right:
            xs = 1.0 - xs
        # Low-frequency noise only, so the front is a wavy sweep, not swiss-cheese holes.
        noise = rng.random((max(2, h // 32), max(2, w // 32))).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)
        self.dissolve = np.clip(0.88 * xs + 0.12 * noise, 0, 1).astype(np.float32)
        self.p0 = _sample(self.mask0, n, rng)
        ix = np.clip(np.rint(self.p0[:, 0]).astype(np.int32), 0, w - 1)
        iy = np.clip(np.rint(self.p0[:, 1]).astype(np.int32), 0, h - 1)
        if self.frozen is not None:
            cols = self.frozen[iy, ix].astype(np.float32)
        else:
            cols = np.full((n, 3), 80.0, np.float32)
        cols = cols * rng.uniform(0.82, 1.12, (n, 1))
        ash = rng.random(n) < 0.18
        cols[ash] = cols[ash] * 0.40 + np.array([38, 36, 42], np.float32)
        self.cols = np.clip(cols, 0, 255)
        # Release time along the sweep + a little jitter, so the front takes most of the 2 s.
        u = self.dissolve[iy, ix] * 0.72 + rng.uniform(-0.06, 0.08, n)
        self.u = np.clip(u, 0.0, 0.78).astype(np.float32)
        side = -1.0 if from_right else 1.0          # drift toward the already-dissolved side, and up
        self.v = np.c_[
            side * rng.uniform(0.22, 0.85, n) * w * 0.42 + rng.normal(0, w * 0.05, n),
            rng.uniform(-0.85, -0.22, n) * h * 0.50 + rng.normal(0, h * 0.04, n),
        ].astype(np.float32)
        self.ph = rng.uniform(0, 2 * np.pi, (n, 2)).astype(np.float32)
        self.wf = rng.uniform(6.0, 14.0, (n, 2)).astype(np.float32)
        self.amp = (rng.uniform(0.4, 1.2, (n, 2)) * np.array([w, h]) * 0.018).astype(np.float32)
        self.sz = rng.choice(np.array([1, 2, 3]), n, p=[0.55, 0.35, 0.10]).astype(np.int32)
        self.life = rng.uniform(0.50, 0.88, n).astype(np.float32)   # visibility after release, in t units

    def _snap_vis_map(self, t):
        """Soft residual of the frozen person: 1 where not yet dissolved.

        Each pixel starts fading when t reaches its release time (same sweep as
        the flakes) and is gone ~0.10 later, so the body doesn't pop off in chunks.
        """
        fade_t = self.dissolve * 0.72
        return np.clip(1.0 - (t - fade_t) / 0.10, 0, 1) * self.mask0

    def _snap_particles(self, t):
        e = np.clip(t - self.u, 0, None)
        # Positions are a closed-form function of normalized t, so reverse is exact.
        turb = self.amp * np.sin(self.wf * e[:, None] + self.ph)
        pos = self.p0 + self.v * e[:, None] + turb
        fade = np.clip(1.0 - e / self.life, 0, 1)
        # Don't show a flake before it is released; a short ramp at the start of its life.
        born = np.clip(e / 0.03, 0, 1)
        alpha = fade * fade * born
        # Shrink as they fade.
        sizes = np.maximum(1, np.rint(self.sz * (0.45 + 0.55 * fade))).astype(np.int32)
        # Short streak along the drift for the faster flakes (dust, not squares).
        speed = np.linalg.norm(self.v, axis=1) + 1e-6
        st = np.sign(self.v) * np.clip(np.rint(self.v / speed[:, None] * (1 + (self.sz >= 2))[:, None]), -2, 2)
        st = st.astype(np.int32)
        st[alpha < 0.05] = 0
        return pos, alpha.astype(np.float32), sizes, st

    def _splat_flakes(self, out, xy, cols, alphas, sizes, streaks):
        """Vectorized 1–3 px flake splat (add.at); fine dust, not blocky tiles."""
        h, w = out.shape[:2]
        acc = np.zeros((h, w, 3), np.float32)
        wgt = np.zeros((h, w), np.float32)

        def stamp(xs, ys, a, c):
            vis = (a > 0.03) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if not np.any(vis):
                return
            ys, xs, a, c = ys[vis], xs[vis], a[vis], c[vis]
            np.add.at(wgt, (ys, xs), a)
            np.add.at(acc[:, :, 0], (ys, xs), c[:, 0] * a)
            np.add.at(acc[:, :, 1], (ys, xs), c[:, 1] * a)
            np.add.at(acc[:, :, 2], (ys, xs), c[:, 2] * a)

        xs = np.rint(xy[:, 0]).astype(np.int32)
        ys = np.rint(xy[:, 1]).astype(np.int32)
        stamp(xs, ys, alphas, cols)
        big = sizes >= 2
        if np.any(big):
            stamp(xs[big] + 1, ys[big], alphas[big] * 0.85, cols[big])
            stamp(xs[big], ys[big] + 1, alphas[big] * 0.85, cols[big])
        huge = sizes >= 3
        if np.any(huge):
            stamp(xs[huge] + 1, ys[huge] + 1, alphas[huge] * 0.7, cols[huge])
            stamp(xs[huge] - 1, ys[huge], alphas[huge] * 0.5, cols[huge])
        if streaks is not None:
            stamp(xs + streaks[:, 0], ys + streaks[:, 1], alphas * 0.6, cols)
        m = np.clip(wgt, 0, 1)[..., None]
        nz = wgt > 1e-6
        avg = np.zeros_like(out)
        avg[nz] = acc[nz] / wgt[nz][:, None]
        return out * (1.0 - m) + avg * m

    def _snap(self, frame, mask, plate, t):
        src = self.frozen if self.frozen is not None else frame
        vis = self._snap_vis_map(t)
        # Don't run vis through _comp's 0.25 mask floor — that would eat the soft front.
        a = vis[..., None]
        out = src * a + plate * (1.0 - a)
        pos, alpha, sizes, streaks = self._snap_particles(t)
        if float(alpha.max(initial=0.0)) > 0.03:
            out = self._splat_flakes(out, pos, self.cols, alpha, sizes, streaks)
        return np.clip(out, 0, 255)


def clone_layout(bbox, w):
    """(dx, scale) per clone: 2 at arm's length either side, plus 2 smaller ones further out when they fit."""
    x0, y0, x1, y1 = bbox; bw = max(40, x1 - x0); cx = (x0 + x1) / 2
    lay = [(-bw * 0.85, 0.92), (bw * 0.85, 0.92)]
    far = bw * 1.55
    if cx - far > w * 0.08 and cx + far < w * 0.92: lay = [(-far, 0.8), (far, 0.8)] + lay   # far ones drawn first (behind)
    return lay

def render_clones(frame, mask, layout, bbox, alphas=None, scales=None):
    """Composite shifted/scaled copies of the live cutout behind the real person (who stays on top).
    Works on the person's bounding-box crop only (resize + paste), so 4 clones stay cheap at 720p."""
    h, w = mask.shape; out = frame.copy()
    pad = 30; x0, y0 = max(0, int(bbox[0]) - pad), max(0, int(bbox[1]) - pad)
    x1, y1 = min(w, int(bbox[2]) + pad), h                       # down to the bottom edge (body is cut by the frame)
    cf, cm = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    cx, by = (bbox[0] + bbox[2]) / 2, y1
    for i, (dx, sc) in enumerate(layout):
        a = 1.0 if alphas is None else alphas[i]
        if a <= 0.01: continue
        s = sc * (1.0 if scales is None else scales[i])
        nw, nh = max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))
        # scale about the person's bottom-center, then shift by dx
        X0 = int(round(cx + dx - (cx - x0) * s)); Y0 = int(round(by - (by - y0) * s))
        ax0, ay0, ax1, ay1 = max(0, X0), max(0, Y0), min(w, X0 + nw), min(h, Y0 + nh)
        if ax1 <= ax0 or ay1 <= ay0: continue
        rf = cv2.resize(cf, (nw, nh), interpolation=cv2.INTER_LINEAR)[ay0 - Y0:ay1 - Y0, ax0 - X0:ax1 - X0]
        rm = cv2.resize(cm, (nw, nh), interpolation=cv2.INTER_LINEAR)[ay0 - Y0:ay1 - Y0, ax0 - X0:ax1 - X0, None] * a
        o = out[ay0:ay1, ax0:ax1]; out[ay0:ay1, ax0:ax1] = o + (rf - o) * rm
    ml = mask[y0:y1, x0:x1, None]; o = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = o + (cf - o) * ml                        # the real you on top
    return out

def draw_charge_ring(img, anchor_px, progress, kind):
    """Small charging ring near the hand. Drawn on the preview only by default."""
    if progress <= 0 or anchor_px is None: return img
    x, y = int(anchor_px[0]), int(anchor_px[1]); r = 16
    col = RING_COLOR.get(kind, (255, 255, 255))
    cv2.circle(img, (x, y - 30), r, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.ellipse(img, (x, y - 30), (r, r), -90, 0, 360 * progress, col, 3, cv2.LINE_AA)
    return img
