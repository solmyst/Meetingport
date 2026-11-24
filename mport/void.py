"""Gojo's Domain Expansion: Infinite Void.

A persistent animated backdrop. The person stays fully visible (composited with the
segmentation mask); only the background becomes a cosmic void. Stars, noise, and
coordinate grids are precomputed so a 1280x720 frame stays well under ~15 ms on CPU.
"""
import cv2, numpy as np

def _smooth(a, b, t):
    x = float(np.clip((t - a) / (b - a), 0, 1))
    return x * x * (3 - 2 * x)

def _fbm(h, w, rng, octaves=5, base=16):
    acc = np.zeros((h, w), np.float32); amp, tot = 1.0, 0.0
    for i in range(octaves):
        gh, gw = max(2, base << i), max(2, int(base * w / h) << i)
        n = rng.random((min(gh, h), min(gw, w))).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += amp * n; tot += amp; amp *= 0.52
    acc /= tot
    return cv2.normalize(acc, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

# BGR palettes
_SPACE_CORE = np.array([30, 8, 14], np.float32)
_SPACE_EDGE = np.array([3, 1, 2], np.float32)
_NEB_CYAN = np.array([255, 210, 90], np.float32)
_NEB_VIOLET = np.array([200, 50, 170], np.float32)
_NEB_MAGENTA = np.array([180, 40, 255], np.float32)
_NEB_WHITE = np.array([255, 245, 235], np.float32)
_RING = np.array([255, 230, 210], np.float32)
_RING_CYAN = np.array([255, 200, 80], np.float32)
_RIM = np.array([255, 240, 220], np.float32)
_COL_NEB1 = (0.65 * _NEB_VIOLET + 0.45 * _NEB_CYAN + 0.18 * _NEB_WHITE).astype(np.float32)
_COL_NEB2 = (0.65 * _NEB_MAGENTA + 0.35 * _NEB_CYAN + 0.14 * _NEB_WHITE).astype(np.float32)

_CACHE = {}

def get_void(shape, seed=11):
    key = (int(shape[0]), int(shape[1]), int(seed))
    v = _CACHE.get(key)
    if v is None:
        _CACHE[key] = v = InfiniteVoid(shape, seed=seed)
    return v

def _bbox_lq(mask, q):
    """Person bbox from a 1/q mask (cheap). Returns full-res (x0, y0, x1, y1)."""
    h, w = mask.shape[:2]
    lh, lw = max(1, h // q), max(1, w // q)
    sm = cv2.resize(mask, (lw, lh), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(sm > 0.4)
    if len(xs) == 0:
        return w // 3, h // 4, 2 * w // 3, h
    return int(xs.min() * q), int(ys.min() * q), min(w, int(xs.max() * q + q)), min(h, int(ys.max() * q + q))

def head_point(mask, bbox=None):
    """A point around the head: horizontal center, upper ~18% of the silhouette."""
    if bbox is None: bbox = _bbox_lq(mask, 4)
    x0, y0, x1, y1 = bbox
    return (0.5 * (x0 + x1), y0 + 0.18 * (y1 - y0))


class InfiniteVoid:
    """Precomputed starfield + low-res swirling nebula, composited under the person."""

    def __init__(self, shape, seed=11):
        self.h, self.w = int(shape[0]), int(shape[1])
        h, w = self.h, self.w
        self.q = 4
        self.lh, self.lw = max(1, h // self.q), max(1, w // self.q)
        self.mq = 2
        self.mh, self.mw = max(1, h // self.mq), max(1, w // self.mq)
        rng = np.random.default_rng(seed)
        self.noise_n = 512
        self.noise = _fbm(self.noise_n, self.noise_n, rng, octaves=5, base=8)
        self.noise2 = _fbm(self.noise_n, self.noise_n, rng, octaves=4, base=6)
        yy, xx = np.indices((self.lh, self.lw), dtype=np.float32)
        self.lq_x, self.lq_y = xx, yy
        # Reusable buffers (avoid per-frame allocations on the hot path).
        self.neb = np.empty((self.lh, self.lw, 3), np.float32)
        self.mid = np.empty((self.mh, self.mw, 3), np.float32)
        self.mid_fr = np.empty((self.mh, self.mw, 3), np.float32)
        self.out = np.empty((h, w, 3), np.float32)
        self._r = np.empty((self.lh, self.lw), np.float32)
        self._ang = np.empty((self.lh, self.lw), np.float32)
        self._build_stars(rng)
        self._build_streaks(rng)
        self._build_portal(rng)
        self._caption = "Domain Expansion: Infinite Void"
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._scale = 0.68 if w >= 1100 else (0.56 if w >= 800 else 0.46)
        self._thick = 1
        (tw, th), base = cv2.getTextSize(self._caption, self._font, self._scale, self._thick)
        self._tw, self._th, self._base = tw, th, base
        self._cx = (w - tw) // 2
        self._cy = int(h * 0.972)
        self._morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _build_stars(self, rng):
        w, h = self.mw, self.mh
        n0 = 650
        self.s0_x = rng.random(n0, dtype=np.float32)
        self.s0_y = rng.random(n0, dtype=np.float32)
        self.s0_b = rng.uniform(0.25, 0.7, n0).astype(np.float32)
        self.s0_ph = rng.uniform(0, 2 * np.pi, n0).astype(np.float32)
        self.s0_fr = rng.uniform(0.7, 2.2, n0).astype(np.float32)
        self.s0_vx = rng.uniform(-0.003, 0.003, n0).astype(np.float32)
        self.s0_vy = rng.uniform(-0.002, 0.002, n0).astype(np.float32)
        pal = np.array([[255, 245, 235], [255, 220, 180], [230, 180, 255]], np.float32)
        self.s0_c = pal[rng.integers(0, 3, n0)]
        n1 = 240
        self.s1_x = rng.random(n1, dtype=np.float32)
        self.s1_y = rng.random(n1, dtype=np.float32)
        self.s1_b = rng.uniform(0.45, 1.0, n1).astype(np.float32)
        self.s1_ph = rng.uniform(0, 2 * np.pi, n1).astype(np.float32)
        self.s1_fr = rng.uniform(0.5, 1.8, n1).astype(np.float32)
        self.s1_vx = rng.uniform(-0.008, 0.008, n1).astype(np.float32)
        self.s1_vy = rng.uniform(-0.005, 0.005, n1).astype(np.float32)
        pal = np.array([[255, 250, 245], [255, 210, 120], [220, 140, 255]], np.float32)
        self.s1_c = pal[rng.integers(0, 3, n1)]
        self.s1_sz = rng.integers(1, 3, n1).astype(np.int32)
        n2 = 80
        self.s2_ang = rng.uniform(0, 2 * np.pi, n2).astype(np.float32)
        self.s2_r = rng.uniform(0.08, 0.75, n2).astype(np.float32)
        self.s2_pull = rng.uniform(0.018, 0.055, n2).astype(np.float32)
        self.s2_spin = rng.uniform(-0.08, 0.08, n2).astype(np.float32)
        self.s2_b = rng.uniform(0.7, 1.2, n2).astype(np.float32)
        self.s2_ph = rng.uniform(0, 2 * np.pi, n2).astype(np.float32)
        self.s2_fr = rng.uniform(0.8, 2.5, n2).astype(np.float32)
        pal = np.array([[255, 255, 250], [255, 230, 160], [240, 200, 255]], np.float32)
        self.s2_c = pal[rng.integers(0, 3, n2)]
        self.s2_sz = rng.integers(1, 3, n2).astype(np.int32)
        self._diag = 0.5 * float(np.hypot(w, h))

    def _build_streaks(self, rng):
        n = 32
        self.st_ang = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.st_ph = rng.uniform(0, 1, n).astype(np.float32)
        self.st_spd = rng.uniform(0.18, 0.45, n).astype(np.float32)
        self.st_len = rng.uniform(0.08, 0.22, n).astype(np.float32)
        self.st_w = rng.integers(1, 3, n)
        pal = np.array([[255, 250, 230], [255, 210, 120], [230, 160, 255]], np.float32)
        self.st_c = pal[rng.integers(0, 3, n)]
        self.st_b = rng.uniform(0.35, 0.85, n).astype(np.float32)

    def _build_portal(self, rng):
        """Bake a high-quality event-horizon sprite (BGR + alpha) at init."""
        S = 384
        c = (S // 2, S // 2)
        r = int(S * 0.28)
        bgr = np.zeros((S, S, 3), np.float32)
        # Violet / indigo bloom
        cv2.circle(bgr, c, int(r * 2.15), (50, 10, 28), -1, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 1.70), (90, 18, 50), -1, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 1.35), (130, 28, 70), -1, cv2.LINE_AA)
        # Event horizon (near-black disk)
        cv2.circle(bgr, c, int(r * 1.08), (10, 3, 8), -1, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 0.96), (2, 0, 3), -1, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 0.55), (8, 3, 6), -1, cv2.LINE_AA)
        # Rings
        cv2.circle(bgr, c, int(r * 1.05), (200, 110, 70), 3, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 0.98), (255, 240, 225), 4, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 0.90), (220, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(bgr, c, int(r * 0.84), (160, 80, 60), 1, cv2.LINE_AA)
        # Horizontal iris slit
        cv2.ellipse(bgr, c, (int(r * 0.62), max(1, int(r * 0.065))), 0, 0, 360, (190, 100, 80), 1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(bgr, (0, 0), 4.0)
        core = bgr * 0.75 + glow * 1.15
        # Alpha: bloom falls off, disk is solid
        yy, xx = np.indices((S, S), dtype=np.float32)
        dist = np.sqrt((xx - S / 2) ** 2 + (yy - S / 2) ** 2)
        alpha = np.clip(1.2 - dist / (r * 2.05), 0, 1)
        alpha = np.where(dist <= r * 1.08, np.maximum(alpha, 0.96), alpha * alpha)
        self.portal_bgr = np.clip(core, 0, 255).astype(np.float32)
        self.portal_a = alpha.astype(np.float32)
        self.portal_r = r          # sprite radius in sprite pixels
        self.portal_S = S
        # Tick marks as a separate rotating layer
        ticks = np.zeros((S, S, 3), np.float32)
        ta = np.zeros((S, S), np.float32)
        for i in range(8):
            a = i * (np.pi / 4)
            p0 = (int(S / 2 + np.cos(a) * r * 0.98), int(S / 2 + np.sin(a) * r * 0.98))
            p1 = (int(S / 2 + np.cos(a) * r * 1.24), int(S / 2 + np.sin(a) * r * 1.24))
            cv2.line(ticks, p0, p1, (210, 140, 100), 2, cv2.LINE_AA)
        ticks = cv2.GaussianBlur(ticks, (0, 0), 1.2)
        ta = np.clip(ticks.max(axis=2) / 210.0, 0, 1)
        # Prebake 16 rotation frames so the portal isn't warpAffine'd every frame.
        self.tick_n = 16
        self.tick_bgr = []
        self.tick_a = []
        for i in range(self.tick_n):
            M = cv2.getRotationMatrix2D((S / 2, S / 2), i * (360 / self.tick_n), 1.0)
            self.tick_bgr.append(cv2.warpAffine(ticks, M, (S, S)))
            self.tick_a.append(cv2.warpAffine(ta, M, (S, S)))

    def _nebula(self, hx, hy, t):
        """Low-res swirling nebula + indigo space, flowing toward (hx, hy) in full-res pixels."""
        q, lw, lh = self.q, self.lw, self.lh
        cx, cy = hx / q, hy / q
        dx = self.lq_x - cx; dy = self.lq_y - cy
        r = np.sqrt(dx * dx + dy * dy, out=self._r); r += 1e-3
        ang = np.arctan2(dy, dx, out=self._ang)
        N = self.noise_n
        flow = t * 0.42
        u = ((ang * (0.5 / np.pi) + 0.5 + 0.07 * np.sin(ang * 3.0 + t * 0.35)) * (N - 1)).astype(np.float32)
        v = ((np.log(r + 6.0) * 70.0 - flow * 55.0) % (N - 1)).astype(np.float32)
        u2 = ((ang * (0.5 / np.pi) + 0.5 + 0.175) * (N - 1)).astype(np.float32)
        v2 = ((np.log(r + 3.5) * 48.0 - flow * 36.0 + 19.0) % (N - 1)).astype(np.float32)
        n1 = cv2.remap(self.noise, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n2 = cv2.remap(self.noise2, u2, v2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        arms = 0.5 + 0.5 * np.sin(ang * 4.0 + np.log(r + 1.0) * 2.0 - t * 0.7)
        rmax = float(np.hypot(max(cx, self.lw - cx), max(cy, self.lh - cy))) + 1e-3
        fall = np.clip(1.0 - (r / (rmax * 0.95)), 0, 1)
        fall *= np.sqrt(fall)            # ~ **0.65 without a slow pow
        hole = np.clip((r - 6.0) * 0.10, 0, 1)
        d1 = (n1 * n1) * (0.35 + 0.75 * arms) * fall * hole
        d2 = (n2 * n2) * fall * hole
        glow = np.exp(-(r / (rmax * 0.45)) ** 2)
        space = _SPACE_EDGE + (_SPACE_CORE - _SPACE_EDGE) * glow[..., None]
        neb = d1[..., None] * _COL_NEB1 + d2[..., None] * _COL_NEB2
        neb += (n1 * arms * fall * 0.40)[..., None] * _NEB_CYAN
        np.clip(space + neb, 0, 255, out=self.neb)
        return self.neb

    def _stamp_stars(self, canvas, t, hx, hy):
        mh, mw = self.mh, self.mw
        x = (self.s0_x + self.s0_vx * t) % 1.0
        y = (self.s0_y + self.s0_vy * t) % 1.0
        tw = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(t * self.s0_fr + self.s0_ph))
        b = self.s0_b * tw
        xs = (x * (mw - 1)).astype(np.int32); ys = (y * (mh - 1)).astype(np.int32)
        canvas[ys, xs] += self.s0_c * b[:, None]
        x = (self.s1_x + self.s1_vx * t) % 1.0
        y = (self.s1_y + self.s1_vy * t) % 1.0
        tw = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(t * self.s1_fr + self.s1_ph)) ** 1.4
        b = self.s1_b * tw
        xs = (x * (mw - 1)).astype(np.int32); ys = (y * (mh - 1)).astype(np.int32)
        canvas[ys, xs] += self.s1_c * b[:, None]
        xs2 = np.clip(xs + 1, 0, mw - 1); ys2 = np.clip(ys + 1, 0, mh - 1)
        big = self.s1_sz >= 2
        canvas[ys2[big], xs[big]] += self.s1_c[big] * (0.5 * b[big, None])
        canvas[ys[big], xs2[big]] += self.s1_c[big] * (0.5 * b[big, None])
        cx, cy = hx / self.mq, hy / self.mq
        r = (self.s2_r - self.s2_pull * t) % 0.78
        ang = self.s2_ang + self.s2_spin * t
        xs = (cx + np.cos(ang) * r * self._diag).astype(np.int32)
        ys = (cy + np.sin(ang) * r * self._diag).astype(np.int32)
        ok = (xs >= 1) & (xs < mw - 1) & (ys >= 1) & (ys < mh - 1)
        fade = np.clip(r / 0.08, 0, 1) * np.clip((0.78 - r) / 0.08, 0, 1)
        tw = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(t * self.s2_fr + self.s2_ph))
        b = self.s2_b * tw * fade
        xs, ys, b, c, sz = xs[ok], ys[ok], b[ok], self.s2_c[ok], self.s2_sz[ok]
        canvas[ys, xs] += c * b[:, None]
        big = sz >= 2
        canvas[ys[big], np.clip(xs[big] + 1, 0, mw - 1)] += c[big] * (0.6 * b[big, None])
        canvas[np.clip(ys[big] + 1, 0, mh - 1), xs[big]] += c[big] * (0.6 * b[big, None])

    def _streaks(self, canvas, t, hx, hy):
        cx, cy = hx / self.mq, hy / self.mq
        R = self._diag * 1.05
        u = (self.st_ph + t * self.st_spd) % 1.15
        live = (u < 1.0) & (u > 0.02)
        for ang, uu, L, ww, col, br in zip(
            self.st_ang[live], u[live], self.st_len[live], self.st_w[live],
            self.st_c[live], self.st_b[live],
        ):
            r1 = uu * R; r0 = max(0.0, r1 - L * R)
            ca, sa = np.cos(ang), np.sin(ang)
            p0 = (int(cx + ca * r0), int(cy + sa * r0))
            p1 = (int(cx + ca * r1), int(cy + sa * r1))
            vis = br * min(1.0, uu / 0.12) * min(1.0, (1.0 - uu) / 0.12)
            cv2.line(canvas, p0, p1, (col * vis).tolist(), int(ww), cv2.LINE_AA)

    def _portal(self, canvas, hx, hy, t):
        """Paste the baked event-horizon sprite behind the head, with rotating ticks."""
        mh, mw = self.mh, self.mw
        rad = max(28.0, min(mh, mw) * 0.22)          # mid-res pixel radius of the disk
        # Sprite is 384px with disk radius portal_r; scale so the disk matches `rad`.
        dest = max(16, int(round(self.portal_S * (rad / (self.portal_r)))))
        dest = dest | 1  # odd, keeps the center on a pixel
        cx, cy = hx / self.mq, hy / self.mq
        x0 = int(round(cx - dest / 2)); y0 = int(round(cy - dest / 2))
        ax0, ay0 = max(0, x0), max(0, y0)
        ax1, ay1 = min(mw, x0 + dest), min(mh, y0 + dest)
        if ax1 <= ax0 or ay1 <= ay0: return
        spr = cv2.resize(self.portal_bgr, (dest, dest), interpolation=cv2.INTER_LINEAR)
        sa = cv2.resize(self.portal_a, (dest, dest), interpolation=cv2.INTER_LINEAR)
        # Rotating ticks (nearest prebaked frame)
        ti = int((t * 0.25 / (2 * np.pi) * self.tick_n)) % self.tick_n
        ticks = cv2.resize(self.tick_bgr[ti], (dest, dest), interpolation=cv2.INTER_LINEAR)
        ta = cv2.resize(self.tick_a[ti], (dest, dest), interpolation=cv2.INTER_LINEAR)
        # Crop to canvas
        sx0, sy0 = ax0 - x0, ay0 - y0
        sx1, sy1 = sx0 + (ax1 - ax0), sy0 + (ay1 - ay0)
        roi = canvas[ay0:ay1, ax0:ax1]
        a = sa[sy0:sy1, sx0:sx1, None]
        roi *= (1.0 - a)
        roi += spr[sy0:sy1, sx0:sx1] * a
        # Ticks additive, slightly on top of the ring
        ta = ta[sy0:sy1, sx0:sx1, None]
        roi += ticks[sy0:sy1, sx0:sx1] * ta
        # A hot highlight walking around the ring
        ha = t * 1.3
        hx_ = int(cx + np.cos(ha) * rad * 0.98)
        hy_ = int(cy + np.sin(ha) * rad * 0.98)
        cv2.circle(canvas, (hx_, hy_), max(2, int(rad * 0.06)), (255, 250, 240), -1, cv2.LINE_AA)

    def _expansion(self, origin, coverage):
        ox, oy = origin
        e = _smooth(0.0, 1.0, coverage) ** 0.85
        q = self.q
        dx = self.lq_x - ox / q; dy = self.lq_y - oy / q
        dist = np.sqrt(dx * dx + dy * dy)
        w, h = self.lw, self.lh
        rmax = max(np.hypot(ox / q, oy / q), np.hypot(ox / q - w, oy / q),
                   np.hypot(ox / q, oy / q - h), np.hypot(ox / q - w, oy / q - h)) * 1.08
        R = e * rmax
        soft = max(2.0, 0.05 * rmax)
        u = np.clip((R + soft - dist) / max(2.0 * soft, 1e-3), 0, 1)
        matte = u * u * (3 - 2 * u)
        vis = float(np.clip(min(coverage / 0.08, (1.0 - coverage) / 0.10), 0, 1))
        sig = max(1.4, 0.016 * rmax)
        ring = np.exp(-((dist - R) / sig) ** 2) * vis
        shock = np.exp(-((dist - R - 2.2 * sig) / (2.4 * sig)) ** 2) * (0.4 * vis)
        return matte, ring, shock

    def _rim_in_crop(self, out, mask, amount, x0, y0, x1, y1):
        """Soft white rim on the silhouette edge only (not the interior), in a padded crop."""
        if amount <= 0.01: return
        mc = mask[y0:y1, x0:x1]
        hh, ww = mc.shape
        if hh < 4 or ww < 4: return
        sh, sw = max(1, hh // 2), max(1, ww // 2)
        sm = cv2.resize(mc, (sw, sh), interpolation=cv2.INTER_AREA)
        bin_ = (sm > 0.5).astype(np.uint8)
        grad = cv2.morphologyEx(bin_, cv2.MORPH_GRADIENT, self._morph_k).astype(np.float32)
        soft = cv2.blur(grad, (3, 3))
        dil = cv2.dilate(bin_, self._morph_k)
        outer = cv2.blur((dil - bin_).astype(np.float32), (5, 5))
        rim = np.clip(soft * 1.3 + outer * 0.45, 0, 1)
        rim = cv2.resize(rim, (ww, hh), interpolation=cv2.INTER_LINEAR)
        out[y0:y1, x0:x1] += rim[..., None] * _RIM * (0.55 * amount)

    def _draw_caption(self, img, alpha):
        if alpha <= 0.01: return img
        a = float(np.clip(alpha, 0, 1))
        x, y = self._cx, self._cy
        tw, th, base = self._tw, self._th, self._base
        pad_x, pad_y = 16, 10
        y0, y1 = max(0, y - th - pad_y), min(self.h, y + base + pad_y)
        x0, x1 = max(0, x - pad_x), min(self.w, x + tw + pad_x)
        roi = img[y0:y1, x0:x1]
        # Fresh 8-bit layer (OpenCV 5 putText requires CV_8U): dark plate + single shadow + text.
        layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        lx, ly = x - x0, y - y0
        cv2.putText(layer, self._caption, (lx + 1, ly + 1), self._font, self._scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(layer, self._caption, (lx, ly), self._font, self._scale, (255, 236, 220), 1, cv2.LINE_AA)
        plate_a = 0.40 * a
        text_a = (layer.max(axis=2).astype(np.float32) / 255.0) * a
        out = roi * (1.0 - plate_a)
        ta = text_a[..., None]
        out = out * (1.0 - ta) + layer.astype(np.float32) * ta
        img[y0:y1, x0:x1] = out
        return img

    def render(self, frame, mask, coverage, clock, origin=None, caption=True, caption_alpha=None):
        """Composite the void under the person.

        coverage: 0..1 opening amount (1 = fully open). clock: seconds, drives animation.
        origin: pixel (x, y) the domain expands from (the hand). caption fades with coverage
        unless caption_alpha is given.
        """
        frame = np.asarray(frame, dtype=np.float32)
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3: m = m[..., 0]
        if m.shape[:2] != (self.h, self.w):
            m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        bb = _bbox_lq(m, self.q)
        hx, hy = head_point(m, bb)
        if origin is None: origin = (hx, hy + 0.25 * (bb[3] - bb[1]))
        ox, oy = float(origin[0]), float(origin[1])
        t = float(clock)

        neb = self._nebula(hx, hy, t)
        cv2.resize(neb, (self.mw, self.mh), dst=self.mid, interpolation=cv2.INTER_LINEAR)
        self._stamp_stars(self.mid, t, hx, hy)
        self._streaks(self.mid, t, hx, hy)
        self._portal(self.mid, hx, hy, t)
        np.clip(self.mid, 0, 255, out=self.mid)

        cov = float(np.clip(coverage, 0, 1))
        if 0.001 < cov < 0.999:
            matte, ring, shock = self._expansion((ox, oy), cov)
            cv2.resize(frame, (self.mw, self.mh), dst=self.mid_fr, interpolation=cv2.INTER_AREA)
            mt = cv2.resize(matte, (self.mw, self.mh), interpolation=cv2.INTER_LINEAR)[..., None]
            rg = cv2.resize(ring, (self.mw, self.mh), interpolation=cv2.INTER_LINEAR)
            sh = cv2.resize(shock, (self.mw, self.mh), interpolation=cv2.INTER_LINEAR)
            # mid = camera * (1-matte) + void * matte + rings  (write back into self.mid)
            self.mid *= mt
            self.mid += self.mid_fr * (1.0 - mt)
            self.mid += rg[..., None] * _RING * 1.6
            self.mid += sh[..., None] * _RING_CYAN * 0.7
            np.clip(self.mid, 0, 255, out=self.mid)

        cv2.resize(self.mid, (self.w, self.h), dst=self.out, interpolation=cv2.INTER_LINEAR)
        # Person over the void, only in a padded bbox (interior pixels stay exact).
        pad = 24
        x0, y0, x1, y1 = bb
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(self.w, x1 + pad), min(self.h, y1 + pad)
        if cov <= 0.001:
            # Domain hasn't appeared yet: the whole frame is still the camera.
            self.out[:] = frame
        else:
            mc = m[y0:y1, x0:x1, None]
            self.out[y0:y1, x0:x1] = self.out[y0:y1, x0:x1] * (1.0 - mc) + frame[y0:y1, x0:x1] * mc
            open_boost = 0.85 if cov >= 0.999 else (0.45 + 0.7 * _smooth(0.08, 0.75, cov))
            self._rim_in_crop(self.out, m, open_boost, x0, y0, x1, y1)

        if caption:
            if caption_alpha is None:
                caption_alpha = 1.0 if cov >= 0.99 else _smooth(0.42, 0.82, cov)
            self._draw_caption(self.out, caption_alpha)
        return np.clip(self.out, 0, 255)
