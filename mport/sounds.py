"""Royalty-free synthesized SFX (generated here with NumPy, no downloaded audio) + non-blocking playback."""
import os, sys, wave, subprocess, numpy as np
SR = 44100
SOUND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sounds")

def _env(n, a=0.01, r=0.3):
    t = np.linspace(0, 1, n); return np.minimum(1, t / a) * np.clip((1 - t) / r, 0, 1) ** 1.5

def _noise(n, rng, cutoff=0.1):
    x = rng.standard_normal(n); y = np.zeros(n); acc = 0.0
    for i in range(n): acc += cutoff * (x[i] - acc); y[i] = acc   # one-pole low-pass
    return y / (np.abs(y).max() + 1e-9)

def synth(kind, rng=None):
    rng = rng or np.random.default_rng(3)
    if kind == "goku":     # rising charged whine + bright zap
        d = 0.9; n = int(SR * d); t = np.arange(n) / SR
        f = 300 * (12 ** (t / d)); ph = 2 * np.pi * np.cumsum(f) / SR
        whine = np.sin(ph + 2 * np.sin(ph * 0.5)) * np.clip(t / 0.35, 0, 1) * (t < 0.42)
        z = np.zeros(n); zs = int(0.38 * SR); zn = n - zs; tz = np.arange(zn) / SR
        z[zs:] = (np.sin(2 * np.pi * 1800 * tz * (1 - tz)) * 0.6 + rng.standard_normal(zn) * 0.4) * np.exp(-tz * 9)
        x = 0.5 * whine + 0.8 * z
    elif kind == "naruto":  # low thump + airy smoke poof
        d = 0.9; n = int(SR * d); t = np.arange(n) / SR
        thump = np.sin(2 * np.pi * (90 - 40 * t) * t) * np.exp(-t * 14)
        x = 0.9 * thump + 0.7 * _noise(n, rng, 0.08) * _env(n, 0.02, 0.9)
    elif kind == "gojo":    # dark chord swell with shimmering partials
        d = 1.6; n = int(SR * d); t = np.arange(n) / SR
        x = sum(np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 5 * t)) for f in (110, 138.6, 164.8, 220))
        x = x / 4 * np.sin(np.pi * t / d) + 0.25 * np.sin(2 * np.pi * 1760 * t) * np.sin(np.pi * t / d) ** 6 * np.sin(2 * np.pi * 9 * t)
    elif kind == "clone":   # three quick "poof"s (one per clone) over a soft whoosh
        d = 1.0; n = int(SR * d); t = np.arange(n) / SR; x = 0.35 * _noise(n, rng, 0.05) * _env(n, 0.05, 0.8)
        for k, t0 in enumerate((0.18, 0.30, 0.42)):
            m = t >= t0; tt = t[m] - t0
            x[m] += (np.sin(2 * np.pi * (140 + 25 * k - 60 * tt) * tt) * 0.9 + rng.standard_normal(m.sum()) * 0.35) * np.exp(-tt * 16)
    elif kind == "kamehameha":  # rising charge hum, then a roaring beam blast
        d = 1.9; n = int(SR * d); t = np.arange(n) / SR; tc = 0.8
        f = 90 * (4 ** np.clip(t / tc, 0, 1)); ph = 2 * np.pi * np.cumsum(f) / SR
        hum = (np.sin(ph) + 0.4 * np.sin(2 * ph + np.sin(ph * 0.25))) * np.clip(t / tc, 0, 1) ** 2 * (t < tc)
        roar = (_noise(n, rng, 0.15) + 0.4 * np.sign(np.sin(2 * np.pi * 70 * t))) * (t >= tc) * np.exp(-np.clip(t - tc, 0, None) * 1.6)
        x = 0.6 * hum + 0.9 * roar
    elif kind == "snap":  # sharp finger-snap transient + a thin dusty whoosh
        d = 1.4; n = int(SR * d); t = np.arange(n) / SR
        click_n = int(0.012 * SR)
        click = np.zeros(n)
        burst = rng.standard_normal(click_n)
        burst = np.concatenate([[0], np.diff(burst)])          # bright HF click
        burst = burst / (np.abs(burst).max() + 1e-9)
        click[:click_n] = burst * np.hanning(click_n)
        whoosh = _noise(n, rng, 0.2) * np.exp(-t * 3.2) * (t > 0.01)
        # a little body in the whoosh (air moving)
        body = np.sin(2 * np.pi * (90 - 40 * t) * t) * np.exp(-t * 8) * 0.15
        x = 0.95 * click + 0.45 * whoosh + body
    else: raise ValueError(kind)
    x = x / (np.abs(x).max() + 1e-9) * 0.8
    return x.astype(np.float32)

def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x * 32767).astype("<i2").tobytes())

def ensure_sounds():
    os.makedirs(SOUND_DIR, exist_ok=True); paths = {}
    for k in ("goku", "naruto", "gojo", "clone", "kamehameha", "snap"):
        for rev in (False, True):
            p = os.path.join(SOUND_DIR, f"{k}{'_reverse' if rev else ''}.wav")
            if not os.path.exists(p):
                x = synth(k); write_wav(p, x[::-1].copy() if rev else x)
            paths[(k, rev)] = p
    return paths

def play(path):
    """Fire-and-forget. Plays on your speakers only (not into the call's mic)."""
    try:
        if sys.platform == "darwin": subprocess.Popen(["afplay", path])
        elif sys.platform.startswith("win"):
            import winsound; winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            for cmd in (["paplay", path], ["aplay", "-q", path]):
                try: subprocess.Popen(cmd, stderr=subprocess.DEVNULL); break
                except FileNotFoundError: continue
    except Exception as e: print("sound failed:", e)
