#!/bin/bash
# Hands the annotators' VNC screens over to Manage annotators, once (gui/vnc_screens.py).
#
# Afterwards every screen is cutie-screen@N, a service of this account's own systemd that Save in
# Manage annotators enables (new port) and disables (removed port), and that starts at boot. This
# script:
#   1. disables the system tigervncserver@:N screens and their lines in vncserver.users (sudo),
#   2. lets this account's services run without a login, so screens start at boot (sudo),
#   3. stops the older cutie-screen-N screens and starts every port of annotators.json.
# Every screen restarts once, disconnecting whoever is on it; passwords and SSH do not change.
#
# Run it as the account the screens belong to, from SSH or the real monitor (:1), not from a VNC
# screen it is about to restart:
#   scripts/setup_annotator_screens.sh [task root, default workspace/cutie_tasks]
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:-workspace/cutie_tasks}"
USERS_FILE=/etc/tigervnc/vncserver.users

if [ "$(id -u)" = 0 ]; then
  echo "Run this as the screens' account, not as root: it asks for sudo itself." >&2
  exit 1
fi
if [ ! -f "$TASKS/annotators.json" ]; then
  echo "No annotators.json in $TASKS" >&2
  exit 1
fi
display="${DISPLAY:-}"
display="${display##*:}"
display="${display%%.*}"
if [ -n "$display" ] && [ "$display" -ge 2 ] 2>/dev/null; then
  echo "This terminal is on VNC screen :$display, which restarts; run it from SSH or screen :1." >&2
  exit 1
fi

system_screens=$(ls /etc/systemd/system/multi-user.target.wants/ 2>/dev/null \
  | sed -n 's/^tigervncserver@\(:[0-9]*\)\.service$/\1/p' | sort -t: -k2 -n | tr '\n' ' ')
echo "System VNC screens to hand over: ${system_screens:-none}"
echo "Ports in $TASKS/annotators.json:"
.venv/bin/python -c "import json, sys; [print('  ', e['port'], e['name']) for e in
  json.load(open(sys.argv[1]))['annotators']]" "$TASKS/annotators.json"
read -r -p "Every screen restarts once and its viewer is disconnected. Continue? [y/N] " answer
[ "$answer" = y ] || [ "$answer" = Y ] || exit 1

for screen in $system_screens; do
  sudo systemctl disable --now "tigervncserver@$screen.service"
  sudo sed -i "/^$screen=$USER\$/d" "$USERS_FILE"
done
sudo loginctl enable-linger "$USER"

systemctl --user stop 'cutie-screen-*.service' 2>/dev/null || true
sleep 1
env -u PYTHONPATH .venv/bin/python -m gui.vnc_screens sync --tasks "$TASKS"
