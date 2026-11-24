#!/usr/bin/env bash
# macOS / Linux: ask for the access password, create a Python 3.10–3.12 venv, install, and launch.
set -euo pipefail
cd "$(dirname "$0")"

version_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1
}

pick_python() {
  if [ -n "${PYTHON:-}" ]; then
    if version_ok "$PYTHON"; then
      printf '%s\n' "$PYTHON"
      return 0
    fi
    echo "PYTHON=$PYTHON is not Python 3.10, 3.11, or 3.12." >&2
    return 1
  fi
  local c
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && version_ok "$c"; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

if ! PY=$(pick_python); then
  cat >&2 <<'EOF'
Meetingport needs Python 3.10, 3.11, or 3.12.
MediaPipe does not publish a wheel for every newer Python.

  macOS:   brew install python@3.12
  Linux:   install python3.12 and the matching venv package
           (Debian/Ubuntu: sudo apt install python3.12 python3.12-venv)

Then run ./run.sh again.
Point at an interpreter with:  PYTHON=/path/to/python3.12 ./run.sh
EOF
  exit 1
fi

ok=0
pw=
for _attempt in 1 2 3; do
  if ! read -r -s -p "Meetingport access password: " pw; then
    echo
    echo "Could not read a password." >&2
    exit 1
  fi
  echo
  if MEETINGPORT_PASSWORD="$pw" "$PY" -m mport.access --check; then
    ok=1
    break
  fi
  echo "Wrong password." >&2
  pw=
done
if [ "$ok" -ne 1 ]; then
  echo "Access denied. Ask the person who shared Meetingport for the password." >&2
  exit 1
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Creating a virtual environment with $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
  if ! "$PY" -m venv .venv; then
    echo "Could not create .venv. On Debian/Ubuntu install the venv package, for example: sudo apt install python3.12-venv" >&2
    exit 1
  fi
fi
if ! version_ok "$VENV_PY"; then
  echo "The existing .venv is not Python 3.10, 3.11, or 3.12. Remove it with: rm -rf .venv" >&2
  exit 1
fi

STAMP=".venv/.deps-ok"
if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ]; then
  echo "Installing dependencies"
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
  touch "$STAMP"
fi

CFG="config.json"
if [ ! -f "$CFG" ]; then
  if [ -f config.example.json ]; then
    cp config.example.json config.json
    echo "Created config.json from config.example.json"
  else
    echo "config.json is missing." >&2
    exit 1
  fi
fi

echo "Starting Meetingport"
export MEETINGPORT_PASSWORD="$pw"
unset pw
exec "$VENV_PY" meetingport.py --config "$CFG" "$@"
