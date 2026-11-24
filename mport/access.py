"""Access password for installing Meetingport and for opening a live camera.

The password is not stored in this repository. SALT and EXPECTED are a scrypt
hash. The maintainer shares the password with people who should use the app.

Rotate it with:  python -m mport.access --hash
Then paste the printed SALT and EXPECTED lines over the constants below.
"""
import getpass
import hashlib
import hmac
import os
import secrets
import sys

SCRYPT_N = 2 ** 14
SALT = bytes.fromhex("6e51a9afd631ed8cdea360c8215eceea")
EXPECTED = bytes.fromhex("689aa84608da47514486731a9223a53a0064a9dc296584dd7f322d8448992920")


def hash_password(password, salt=None):
    """Return (salt_bytes, digest_bytes) for a new password."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=8, p=1, dklen=32)
    return salt, digest


def verify(password):
    """True when password matches the maintainer's access password."""
    if not isinstance(password, str) or password == "":
        return False
    digest = hashlib.scrypt(password.encode("utf-8"), salt=SALT, n=SCRYPT_N, r=8, p=1, dklen=32)
    return hmac.compare_digest(digest, EXPECTED)


def needs_password(cfg):
    """Live cameras and the virtual camera ask for the password. Video files do not."""
    source = str(cfg.get("source", "0"))
    if source.isdigit():
        return True
    return bool(cfg.get("virtualcam"))


def ensure_access(cfg):
    """Exit unless this launch may open a camera. Video-file runs return immediately."""
    if not needs_password(cfg):
        return
    env = os.environ.get("MEETINGPORT_PASSWORD")
    if env is not None:
        if verify(env):
            return
        sys.exit("Wrong password.")
    if not sys.stdin.isatty():
        sys.exit(
            "Meetingport needs an access password. Ask the person who shared it with you, "
            "then run again from a terminal."
        )
    for _attempt in range(3):
        entered = getpass.getpass("Meetingport access password: ")
        if verify(entered):
            return
        print("Wrong password.", file=sys.stderr)
    sys.exit("Access denied. Ask the person who shared Meetingport for the password.")


def _hash_cli():
    first = getpass.getpass("New access password: ")
    second = getpass.getpass("Repeat: ")
    if not first or first != second:
        sys.exit("Passwords did not match.")
    salt, digest = hash_password(first)
    print("Replace SALT and EXPECTED in mport/access.py with:")
    print(f"SALT = bytes.fromhex({salt.hex()!r})")
    print(f"EXPECTED = bytes.fromhex({digest.hex()!r})")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--check"]:
        sys.exit(0 if verify(os.environ.get("MEETINGPORT_PASSWORD", "")) else 1)
    if argv == ["--hash"]:
        _hash_cli()
        return
    sys.exit("usage: python -m mport.access --check|--hash")


if __name__ == "__main__":
    main()
