"""Kawarimi log (or a user PNG) composited onto the empty room after a Naruto vanish.

No copied anime assets: the default log is generated in NumPy/OpenCV (bark, growth
rings, branch stubs). A custom PNG with alpha can be used instead.
"""
import os
import cv2
import numpy as np

LOG_HEIGHT_FRAC = 0.50      # of the person bbox height (~torso)
LOG_ASPECT = 0.42           # unrotated width / height
TILT_DEG = -13.0            # slight clockwise lean (top to the right), planted at the base
MODES = ("log", "none", "image")


def person_bbox(mask):
    """(x0, y0, x1, y1) from a person mask, or None if empty."""
    if mask is None: return None
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def log_placement(bbox, frame_shape, scale=1.0, tilt_deg=TILT_DEG):
    """Where the log sits, from the last person bounding box.

    The log is about torso-height, planted at the bottom-middle of the bbox, slightly tilted.
    Returns a dict of floats: cx, bottom, width, height, tilt_deg.
    """
    h, w = frame_shape[:2]
    if bbox is None:
        x0, y0, x1, y1 = w // 3, h // 4, 2 * w // 3, int(h * 0.92)
    else:
        x0, y0, x1, y1 = bbox
    ph = max(8.0, float(y1 - y0))
    scale = float(scale) if scale and scale > 0 else 1.0
    log_h = ph * LOG_HEIGHT_FRAC * scale
    if log_h > h * 0.85:                                     # huge scale still mostly fits
        log_h = h * 0.85
    log_w = log_h * LOG_ASPECT
    cx = (float(x0) + float(x1)) / 2.0
    bottom = float(np.clip(y1, log_h * 0.2, h - 1))
    # If the person was cut off by the frame, lift the log so its base is visible.
    margin = max(4.0, log_w * 0.18)
    if bottom > h - margin:
        bottom = float(h - margin)
    cx = float(np.clip(cx, log_w * 0.35, w - log_w * 0.35))
    return {"cx": cx, "bottom": bottom, "width": log_w, "height": log_h, "tilt_deg": float(tilt_deg)}


def generate_procedural_log(height, width, seed=11):
    """BGRA uint8 standing log: bark cylinder, growth-ring cut end, two branch stubs."""
    h, w = int(max(32, round(height))), int(max(16, round(width)))
    scale = 3                                                # render high-res, then downscale
    pad_x, pad_y = int(round(w * 0.30)), int(round(h * 0.03))  # room for stubs and the cap lip
    H, W = (h + pad_y) * scale, (w + 2 * pad_x) * scale
    content_w = w * scale                                    # cylinder is sized to the requested width
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = (W - 1) / 2.0
    cap_h = (h * scale) * 0.16
    cap_cy, cap_ry = cap_h * 0.55, cap_h * 0.55
    foot_r = content_w * 0.10                                # rounded base, so it isn't a hard crop
    y_body = np.clip((yy - cap_cy) / max(1.0, H - foot_r - cap_cy), 0, 1)
    taper = 1.0 - 0.08 * (1 - y_body)
    r_body = (content_w * 0.38) * taper
    foot = np.clip((yy - (H - foot_r)) / max(1.0, foot_r), 0, 1)
    r_body = r_body * np.sqrt(np.clip(1.0 - foot ** 2, 0, 1))
    cap_rx = float(content_w * 0.38 * (1.0 - 0.08))
    body = (yy >= cap_cy) & (np.abs(xx - cx) <= r_body)
    cap_dist = ((xx - cx) / cap_rx) ** 2 + ((yy - cap_cy) / cap_ry) ** 2
    cap = cap_dist <= 1.0
    alpha = np.zeros((H, W), np.float32)
    alpha[body | cap] = 1.0

    # --- bark ---
    u = np.clip((xx - cx) / np.maximum(r_body, 1.0), -1, 1)
    lambert = np.clip(0.52 + 0.48 * (0.35 - 0.75 * u), 0.22, 1.0)   # light from the left-front
    ang = np.arcsin(np.clip(u, -1, 1))
    grooves = 0.5 + 0.5 * np.sin(ang * 15 + 0.45 * np.sin(yy * 0.07))
    grooves = np.clip(grooves, 0, 1) ** 1.45
    n1 = cv2.resize(rng.normal(0, 1, (max(4, H // 8), max(4, W // 8))).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    n2 = cv2.resize(rng.normal(0, 1, (max(4, H // 3), max(4, W // 3))).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    n = 0.7 * np.clip(n1, -2, 2) + 0.3 * np.clip(n2, -2, 2)
    knot = np.zeros((H, W), np.float32)
    for _ in range(5):
        kx = cx + rng.uniform(-0.65, 0.65) * (content_w * 0.38)
        ky = rng.uniform(cap_h + 0.12 * H, 0.85 * H)
        krx, kry = rng.uniform(content_w * 0.035, content_w * 0.09), rng.uniform(H * 0.012, H * 0.03)
        knot += np.exp(-(((xx - kx) / krx) ** 2 + ((yy - ky) / kry) ** 2))
    tone = np.clip(0.44 + 0.28 * grooves + 0.11 * n - 0.28 * np.clip(knot, 0, 1.5), 0, 1) * lambert
    bark_b = 22 + 68 * tone
    bark_g = 52 + 105 * tone
    bark_r = 78 + 135 * tone
    moss = np.clip(0.16 * (1 - grooves) + 0.07 * (n < -0.6), 0, 0.32) * (yy > cap_cy)
    bark_g = bark_g + 28 * moss
    bark_b = bark_b + 12 * moss
    bgr = np.dstack([bark_b, bark_g, bark_r])

    # --- cut end: pale wood + concentric growth rings + bark rim ---
    pith_x, pith_y = cx + cap_rx * 0.08, cap_cy - cap_ry * 0.05
    ring_r = np.sqrt(((xx - pith_x) / cap_rx) ** 2 + ((yy - pith_y) / cap_ry) ** 2)
    rings = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(ring_r * 22 * np.pi))
    end_n = cv2.resize(rng.normal(0, 1, (max(3, H // 16), max(3, W // 16))).astype(np.float32), (W, H))
    rings = np.clip(rings + 0.08 * end_n, 0, 1)
    end_b = 108 + 72 * rings
    end_g = 158 + 68 * rings
    end_r = 198 + 48 * rings
    rim = np.clip((np.sqrt(np.clip(cap_dist, 0, None)) - 0.80) / 0.20, 0, 1)
    end_b = end_b * (1 - 0.55 * rim) + 35 * rim
    end_g = end_g * (1 - 0.55 * rim) + 65 * rim
    end_r = end_r * (1 - 0.55 * rim) + 85 * rim
    cap_shade = np.clip(0.86 + 0.18 * ((cx - xx) / cap_rx) + 0.10 * ((cap_cy - yy) / cap_ry), 0.7, 1.12)
    end = np.dstack([end_b, end_g, end_r]) * cap_shade[..., None]
    bgr = np.where(cap[..., None], end, bgr)
    # Darken the rounded foot so it reads as sitting on the ground, not a sliced cylinder.
    foot_shade = 1.0 - 0.35 * np.clip(foot, 0, 1) ** 0.7
    bgr = bgr * foot_shade[..., None]

    # --- branch stubs: short capsules sticking out of the bark, cut ends at the tip ---
    stubs = [
        (0.40, -1, 0.28, 0.13, -0.22),   # (y frac, side, length frac of content width, radius frac, upward)
        (0.63,  1, 0.20, 0.10,  0.18),
    ]
    for sy, side, length, rad, ang_up in stubs:
        by = cap_cy + sy * (H - foot_r - cap_cy)
        br = float(content_w * 0.38) * (1.0 - 0.08 * (1 - sy))
        ox = cx + side * br * 0.82
        oy = by
        tip_x = ox + side * (content_w * length)
        tip_y = oy + ang_up * (content_w * length)
        sr = content_w * rad
        alpha, bgr = _stamp_stub(alpha, bgr, ox, oy, tip_x, tip_y, sr, rng)

    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7 * scale)
    alpha = np.clip(alpha * 1.2, 0, 1)
    bgra = np.dstack([np.clip(bgr, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    out_h, out_w = h + pad_y, w + 2 * pad_x
    if (H, W) != (out_h, out_w):
        bgra = cv2.resize(bgra, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return bgra


def _stamp_stub(alpha, bgr, x0, y0, x1, y1, radius, rng):
    """Paint a short bark capsule with a growth-ring cut at the tip. Mutates and returns (alpha, bgr)."""
    H, W = alpha.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = x1 - x0, y1 - y0
    L = max(1e-3, float(np.hypot(dx, dy)))
    ux, uy = dx / L, dy / L
    t = np.clip(((xx - x0) * ux + (yy - y0) * uy) / L, 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    rad = radius * (1.0 - 0.25 * t)
    body = dist <= rad
    tip_d = np.sqrt((xx - x1) ** 2 + (yy - y1) ** 2)
    tip = tip_d <= rad * 0.85
    stamp = body | tip
    if not stamp.any(): return alpha, bgr
    along = t
    n = rng.normal(0, 0.08, stamp.shape)
    tone = np.clip(0.50 + 0.15 * np.sin(along * 18) + n, 0.2, 1.0)
    nx, ny = -uy, ux
    side = np.clip(((xx - px) * nx + (yy - py) * ny) / np.maximum(rad, 1), -1, 1)
    lambert = np.clip(0.55 + 0.45 * side, 0.25, 1.0)
    tone = tone * lambert
    sb, sg, sr_ = 22 + 68 * tone, 52 + 105 * tone, 78 + 135 * tone
    stub = np.dstack([sb, sg, sr_])
    rings = 0.55 + 0.45 * (0.5 + 0.5 * np.sin((tip_d / np.maximum(rad, 1.0)) * 14 * np.pi))
    end = np.dstack([110 + 70 * rings, 158 + 65 * rings, 198 + 45 * rings])
    tip_band = tip_d <= rad * 0.85
    stub = np.where(tip_band[..., None], end, stub)
    bgr = np.where(stamp[..., None], stub, bgr)
    alpha = np.maximum(alpha, stamp.astype(np.float32))
    return alpha, bgr


def resolve_image_path(path):
    if not path: return None
    if os.path.isfile(path): return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt = os.path.join(root, path)
    return alt if os.path.isfile(alt) else None


def load_substitution_image(path, height, width):
    """Load a PNG (alpha used if present), fitted into a height x width BGRA canvas."""
    p = resolve_image_path(path)
    if p is None: return None
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        a = np.full(im.shape[:2] + (1,), 255, np.uint8)
        im = np.concatenate([im, a], axis=2)
    th, tw = int(max(8, round(height))), int(max(8, round(width)))
    ih, iw = im.shape[:2]
    s = min(tw / max(1, iw), th / max(1, ih))
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 4), np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = im
    return canvas


def prepare_substitution(mode, mask, shape, image_path="", scale=1.0):
    """Build (rgba, placement) for a Naruto vanish, or None when mode is 'none' / nothing to draw."""
    mode = (mode or "log").lower()
    if mode not in MODES: mode = "log"
    if mode == "none": return None
    placement = log_placement(person_bbox(mask), shape, scale=scale)
    h, w = placement["height"], placement["width"]
    rgba = None
    if mode == "image":
        rgba = load_substitution_image(image_path, h, w)
        if rgba is None:
            print(f"substitution image not found or unreadable ({image_path!r}); using the procedural log")
    if rgba is None:
        rgba = generate_procedural_log(h, w)
    return rgba, placement


def _rotated_sprite(rgba, tilt_deg):
    """Rotate around the bottom-center so the log stays planted. Returns (rotated BGRA, pivot)."""
    h, w = rgba.shape[:2]
    pad = int(max(h, w) * 0.35) + 2
    padded = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    ph, pw = padded.shape[:2]
    cx, cy = pad + w / 2.0, pad + h
    M = cv2.getRotationMatrix2D((cx, cy), tilt_deg, 1.0)
    rot = cv2.warpAffine(padded, M, (pw, ph), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rot, (cx, cy)


def composite_substitution(plate, rgba, placement, alpha=1.0):
    """Paste the log onto the clean plate with a soft contact shadow and a light color match.

    `plate` is float32 or uint8 BGR. The result has the same dtype / 0..255 range.
    """
    a = float(np.clip(alpha, 0, 1))
    if a <= 0.01 or rgba is None or placement is None: return plate
    src_uint8 = plate.dtype == np.uint8
    out = plate.astype(np.float32).copy()
    H, W = out.shape[:2]
    rot, (px, py) = _rotated_sprite(rgba, placement.get("tilt_deg", TILT_DEG))
    x0 = int(round(placement["cx"] - px))
    y0 = int(round(placement["bottom"] - py))

    # contact shadow: soft ellipse under the planted base, shifted slightly with the lean
    tilt = np.deg2rad(placement.get("tilt_deg", TILT_DEG))
    scx = placement["cx"] - np.sin(tilt) * placement["height"] * 0.08
    scy = placement["bottom"] - 1.0
    sw = max(6.0, placement["width"] * 1.2)
    sh = max(3.0, placement["width"] * 0.30)
    sx0, sx1 = int(np.floor(scx - sw)), int(np.ceil(scx + sw))
    sy0, sy1 = int(np.floor(scy - sh * 1.6)), int(np.ceil(scy + sh * 1.2))
    ax0, ay0, ax1, ay1 = max(0, sx0), max(0, sy0), min(W, sx1), min(H, sy1)
    if ax1 > ax0 and ay1 > ay0:
        yy, xx = np.mgrid[ay0:ay1, ax0:ax1].astype(np.float32)
        ell = np.exp(-(((xx - scx) / (sw * 0.50)) ** 2 + ((yy - scy) / (sh * 0.70)) ** 2))
        ell = cv2.GaussianBlur(ell, (0, 0), max(1.0, sw * 0.08))
        shade = 1.0 - (0.48 * a * np.clip(ell, 0, 1))[..., None]
        out[ay0:ay1, ax0:ax1] *= shade

    rh, rw = rot.shape[:2]
    rx0, ry0, rx1, ry1 = max(0, x0), max(0, y0), min(W, x0 + rw), min(H, y0 + rh)
    if rx1 <= rx0 or ry1 <= ry0:
        return plate.astype(np.uint8) if src_uint8 else plate
    patch = out[ry0:ry1, rx0:rx1]
    spr = rot[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0].astype(np.float32)
    sa = (spr[..., 3] / 255.0) * a
    if sa.max() <= 0.01:
        return plate.astype(np.uint8) if src_uint8 else plate
    log_rgb = spr[..., :3]
    plate_lum = float(np.mean(patch)) if patch.size else 128.0
    log_lum = float((log_rgb * sa[..., None]).sum() / (sa.sum() * 3 + 1e-6))
    if log_lum > 1e-3:
        factor = 0.62 + 0.38 * (plate_lum / log_lum)         # 38% toward the scene
        factor = float(np.clip(factor, 0.55, 1.55))
        log_rgb = np.clip(log_rgb * factor, 0, 255)
    aa = sa[..., None]
    patch[...] = patch * (1 - aa) + log_rgb * aa
    out[ry0:ry1, rx0:rx1] = patch
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8) if src_uint8 else out
