"""Mute the meeting mic when the snap vanish fires, and restore it on reappear.

Default method ("volume") sets the system input volume to 0 (works in every
meeting app, no Accessibility permission on macOS). Optional "hotkey" sends
the meeting app's mute shortcut via System Events (macOS Accessibility must
be granted to the terminal). Failures are logged and never raise.
"""
import shutil
import subprocess
import sys

import cv2
import numpy as np

HOTKEYS = {
    # (key, modifiers) — modifiers are osascript "using" tokens or Windows SendKeys
    "meet":  {"key": "d", "mods": ("command",)},
    "google meet": {"key": "d", "mods": ("command",)},
    "zoom":  {"key": "a", "mods": ("command", "shift")},
    "teams": {"key": "m", "mods": ("command", "shift")},
    "microsoft teams": {"key": "m", "mods": ("command", "shift")},
}


def normalize_app(name):
    n = (name or "meet").strip().lower()
    if n in HOTKEYS:
        return "meet" if n == "google meet" else ("teams" if n == "microsoft teams" else n)
    return "meet"


def macos_get_input_volume_cmd():
    return ["osascript", "-e", "input volume of (get volume settings)"]


def macos_set_input_volume_cmd(value):
    v = int(max(0, min(100, round(float(value)))))
    return ["osascript", "-e", f"set volume input volume {v}"]


def macos_hotkey_cmd(app):
    info = HOTKEYS[normalize_app(app)]
    mods = "using {" + ", ".join(m + " down" for m in info["mods"]) + "}"
    script = f'tell application "System Events" to keystroke "{info["key"]}" {mods}'
    return ["osascript", "-e", script]


def linux_get_mute_cmd():
    return ["pactl", "get-source-mute", "@DEFAULT_SOURCE@"]


def linux_set_mute_cmd(muted):
    return ["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "1" if muted else "0"]


def windows_hotkey_sendkeys(app):
    """PowerShell SendKeys string for the meeting mute shortcut (Win: Ctrl instead of Cmd)."""
    info = HOTKEYS[normalize_app(app)]
    seq = ""
    if "command" in info["mods"] or "control" in info["mods"]:
        seq += "^"
    if "shift" in info["mods"]:
        seq += "+"
    if "alt" in info["mods"]:
        seq += "%"
    seq += info["key"]
    # typelib-free: use WScript.Shell via powershell
    return ["powershell", "-NoProfile", "-Command",
            f"$w = New-Object -ComObject WScript.Shell; $w.SendKeys('{seq}')"]


def parse_input_volume(stdout):
    """Parse `input volume of (get volume settings)` or a similar numeric reply."""
    if stdout is None:
        return None
    s = str(stdout).strip().splitlines()
    if not s:
        return None
    try:
        return int(round(float(s[-1].strip())))
    except ValueError:
        return None


def parse_pactl_mute(stdout):
    s = str(stdout or "").strip().lower()
    if "yes" in s:
        return True
    if "no" in s:
        return False
    return None


class MicController:
    """Mute / restore the system (or meeting-app) microphone. Never raises."""

    def __init__(self, method="volume", meeting_app="meet", runner=None, platform=None, log=print):
        method = (method or "volume").lower().strip()
        if method not in ("volume", "hotkey"):
            method = "volume"
        self.method = method
        self.meeting_app = normalize_app(meeting_app)
        self.runner = runner or _default_run
        self.platform = platform or sys.platform
        self.log = log
        self._prev = None          # previous input volume (0-100) or mute-state
        self.muted = False

    def mute(self):
        if self.muted:
            return True
        ok = self._apply(mute=True)
        if ok:
            self.muted = True
        return ok

    def unmute(self):
        if not self.muted:
            return True
        ok = self._apply(mute=False)
        if ok:
            self.muted = False
            self._prev = None
        return ok

    def close(self):
        if self.muted:
            self.unmute()

    def _run(self, cmd, timeout=3):
        try:
            return self.runner(cmd, timeout=timeout)
        except Exception as e:
            self.log(f"auto-mute: command failed ({e})")
            return None

    def _apply(self, mute):
        sysname = self.platform
        try:
            if self.method == "hotkey":
                return self._hotkey()
            if sysname == "darwin":
                return self._darwin_volume(mute)
            if sysname.startswith("linux"):
                return self._linux_mute(mute)
            if sysname.startswith("win"):
                return self._windows_volume(mute)
            self.log(f"auto-mute: unsupported platform {sysname}")
            return False
        except Exception as e:
            self.log(f"auto-mute: {'mute' if mute else 'unmute'} failed ({e})")
            return False

    def _darwin_volume(self, mute):
        if mute:
            r = self._run(macos_get_input_volume_cmd())
            if r is None:
                return False
            prev = parse_input_volume(getattr(r, "stdout", r))
            if prev is None:
                self.log("auto-mute: could not read macOS input volume")
                return False
            self._prev = prev
            r2 = self._run(macos_set_input_volume_cmd(0))
            return r2 is not None and getattr(r2, "returncode", 0) == 0
        prev = 75 if self._prev is None else self._prev
        r2 = self._run(macos_set_input_volume_cmd(prev))
        return r2 is not None and getattr(r2, "returncode", 0) == 0

    def _linux_mute(self, mute):
        if mute:
            r = self._run(linux_get_mute_cmd())
            prev = parse_pactl_mute(getattr(r, "stdout", "") if r is not None else "")
            self._prev = bool(prev)  # True means already muted; still restore to that
            r2 = self._run(linux_set_mute_cmd(True))
            return r2 is not None and getattr(r2, "returncode", 0) == 0
        # restore previous mute state (default: unmute)
        was_muted = bool(self._prev)
        r2 = self._run(linux_set_mute_cmd(was_muted))
        return r2 is not None and getattr(r2, "returncode", 0) == 0

    def _windows_volume(self, mute):
        # Prefer pycaw (default capture endpoint). Optional nircmd fallback is
        # not used: nircmd's mutesysvolume is the *output* device.
        try:
            return self._pycaw(mute)
        except Exception as e:
            self.log(f"auto-mute: pycaw failed ({e}); install pycaw for Windows mic mute")
            return False

    def _pycaw(self, mute):
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devs = [d for d in AudioUtilities.GetAllDevices()
                if getattr(d, "Flow", None) is not None and "capture" in str(d.Flow).lower()]
        # Fallback: default render is wrong; try GetMicrophone if present, else first capture
        mic = None
        getter = getattr(AudioUtilities, "GetMicrophone", None)
        if callable(getter):
            try:
                mic = getter()
            except Exception:
                mic = None
        if mic is None and devs:
            mic = devs[0]
        if mic is None:
            raise RuntimeError("no capture device")
        iface = mic.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol = iface.QueryInterface(IAudioEndpointVolume)
        if mute:
            try:
                self._prev = float(vol.GetMasterVolumeLevelScalar())
            except Exception:
                self._prev = 1.0
            vol.SetMasterVolumeLevelScalar(0.0, None)
        else:
            prev = 1.0 if self._prev is None else float(self._prev)
            vol.SetMasterVolumeLevelScalar(prev, None)
        return True

    def _hotkey(self):
        # Toggle shortcut — we cannot read mute state. We still send it once
        # per mute()/unmute() and trust the meeting app flipped.
        if self.platform == "darwin":
            cmd = macos_hotkey_cmd(self.meeting_app)
        elif self.platform.startswith("win"):
            cmd = windows_hotkey_sendkeys(self.meeting_app)
        elif self.platform.startswith("linux"):
            # No reliable focused-app keystroke without xdotool/ydotool.
            self.log("auto-mute: hotkey method is not supported on Linux; use mute_method=volume (pactl)")
            return False
        else:
            self.log(f"auto-mute: hotkey unsupported on {self.platform}")
            return False
        r = self._run(cmd)
        if r is None or getattr(r, "returncode", 1) != 0:
            if self.platform == "darwin":
                self.log("auto-mute: hotkey failed (grant Accessibility permission to the "
                         "terminal in System Settings → Privacy & Security → Accessibility)")
            else:
                self.log("auto-mute: hotkey failed")
            return False
        return True


def _default_run(cmd, timeout=3):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def which_pactl():
    return shutil.which("pactl")


def draw_muted_badge(img):
    """Small 'MUTED' chip, intended for the preview window only."""
    if img is None:
        return img
    h, w = img.shape[:2]
    label = "MUTED"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad_x, pad_y = 10, 7
    x1, y0 = w - 12, 12
    x0, y1 = x1 - tw - pad_x * 2, y0 + th + pad_y * 2
    if x0 < 0:
        x0 = 4
    cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 230), 2)
    cv2.putText(img, label, (x0 + pad_x, y1 - pad_y - 1), font, scale, (200, 200, 255), thick, cv2.LINE_AA)
    return img
