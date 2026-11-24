"""Optional mic snap confirmation: a short, broadband, high-frequency click.

The onset detector is pure NumPy and unit-testable. Opening the microphone
(sounddevice) is optional and fails open: if the mic can't be opened, the
app keeps working on gesture alone.
"""
import numpy as np

# Finger snaps are impulsive: a few milliseconds of broadband energy with a
# strong high-frequency (2 kHz+) tail, then silence. Speech and tones are
# longer and more low/mid-heavy.


class SnapOnsetDetector:
    """Streaming onset detector. Feed mono float chunks; process() is True on a snap."""

    def __init__(self, sr=16000, cooldown=0.25, flux_thresh=0.55, hf_ratio_thresh=0.42,
                 min_hf=3e-4, min_crest=4.0, win_ms=10.0, hop_ms=5.0):
        self.sr = int(sr)
        self.cooldown = float(cooldown)
        self.flux_thresh = float(flux_thresh)
        self.hf_ratio_thresh = float(hf_ratio_thresh)
        self.min_hf = float(min_hf)
        self.min_crest = float(min_crest)
        self.win = max(64, int(self.sr * win_ms / 1000.0))
        self.hop = max(32, int(self.sr * hop_ms / 1000.0))
        self.reset()

    def reset(self):
        self.prev_mag = None
        self._buf = np.zeros(0, np.float32)
        self._t = 0.0
        self._cool_until = -1e9

    def process(self, samples, now=None):
        """Consume a mono buffer. Returns True if a snap-like onset is in this block."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return False
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()
        fired = False
        win, hop, sr = self.win, self.hop, self.sr
        while self._buf.size >= win:
            seg = self._buf[:win]
            self._buf = self._buf[hop:]
            t = self._t if now is None else now
            self._t += hop / sr
            if self._analyze(seg, t):
                fired = True
        return fired

    def _mag(self, seg):
        w = np.hanning(seg.size).astype(np.float32)
        return np.abs(np.fft.rfft(seg * w)).astype(np.float32)

    def _analyze(self, seg, t):
        mag = self._mag(seg)
        freqs = np.fft.rfftfreq(seg.size, 1.0 / self.sr)
        hf = freqs >= 2000.0
        e_hf = float(np.mean(mag[hf] ** 2)) if np.any(hf) else 0.0
        e_all = float(np.mean(mag ** 2)) + 1e-12
        hf_ratio = e_hf / e_all
        rms = float(np.sqrt(np.mean(seg * seg))) + 1e-9
        crest = float(np.max(np.abs(seg)) / rms)
        if self.prev_mag is not None and self.prev_mag.shape == mag.shape:
            flux = float(np.mean(np.maximum(mag - self.prev_mag, 0.0)))
            flux_n = flux / (float(np.mean(self.prev_mag)) + 1e-6)
        else:
            flux_n = 0.0
        self.prev_mag = mag
        if t < self._cool_until:
            return False
        if (flux_n >= self.flux_thresh and hf_ratio >= self.hf_ratio_thresh
                and e_hf >= self.min_hf and crest >= self.min_crest):
            self._cool_until = t + self.cooldown
            return True
        return False


class MicSnapListener:
    """Background mic -> SnapOnsetDetector. start() returns False if unavailable."""

    def __init__(self, sr=16000, detector=None):
        self.sr = sr
        self.detector = detector or SnapOnsetDetector(sr=sr)
        self._pending = False
        self._stream = None
        self._err = None

    def start(self):
        try:
            import sounddevice as sd
        except Exception as e:
            self._err = f"sounddevice unavailable ({e})"
            print(f"snap audio: {self._err}; using gesture only")
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=1, blocksize=self.detector.hop * 2,
                dtype="float32", callback=self._cb)
            self._stream.start()
            return True
        except Exception as e:
            self._err = str(e)
            print(f"snap audio: could not open mic ({e}); using gesture only")
            self._stream = None
            return False

    def _cb(self, indata, frames, time_info, status):
        try:
            if self.detector.process(indata[:, 0] if indata.ndim > 1 else indata):
                self._pending = True
        except Exception:
            pass

    def poll(self, now=None):
        """True once per detected click, consumed by the main thread."""
        if self._pending:
            self._pending = False
            return True
        return False

    def close(self):
        st = self._stream
        self._stream = None
        if st is not None:
            try:
                st.stop(); st.close()
            except Exception:
                pass
