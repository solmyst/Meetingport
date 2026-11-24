"""Access password: live camera and virtual camera ask, video files do not."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mport.access import ensure_access, needs_password, verify
from mport.app import main


def test_verify_rejects_wrong_passwords():
    assert verify("") is False
    assert verify("nope") is False
    assert verify("meetingport") is False
    assert verify(None) is False  # type: ignore[arg-type]


def test_needs_password_only_for_live_use():
    assert needs_password({"source": "0", "virtualcam": False}) is True
    assert needs_password({"source": "1", "virtualcam": False}) is True
    assert needs_password({"source": "demo/clips/person.mp4", "virtualcam": True}) is True
    assert needs_password({"source": "demo/clips/person.mp4", "virtualcam": False}) is False


def test_file_source_skips_the_prompt(monkeypatch):
    monkeypatch.delenv("MEETINGPORT_PASSWORD", raising=False)
    ensure_access({"source": "demo/clips/person.mp4", "virtualcam": False})


def test_live_wrong_password_exits(monkeypatch):
    monkeypatch.setenv("MEETINGPORT_PASSWORD", "wrong-password")
    with pytest.raises(SystemExit, match="Wrong password"):
        ensure_access({"source": "0", "virtualcam": False})


def test_live_without_a_terminal_exits(monkeypatch):
    monkeypatch.delenv("MEETINGPORT_PASSWORD", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="access password"):
        ensure_access({"source": "0", "virtualcam": False})


def test_main_file_source_does_not_ask_for_a_password(monkeypatch):
    monkeypatch.delenv("MEETINGPORT_PASSWORD", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--source", "missing-clip.mp4", "--no-preview", "--no-sound", "--plate-seconds", "0"])
    message = str(exc.value).lower()
    assert "password" not in message
    assert "cannot open source" in message


def test_main_live_camera_rejects_wrong_password_before_opening_it(monkeypatch):
    monkeypatch.setenv("MEETINGPORT_PASSWORD", "wrong-password")
    with pytest.raises(SystemExit, match="Wrong password"):
        main(["--source", "0", "--no-preview", "--no-sound"])
