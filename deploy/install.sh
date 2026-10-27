#!/bin/bash
# Installs the CutiePlus cooperation module on this computer (docs/COOPERATION.md). Run it from the
# repository, as the desktop account with sudo:
#
#   sudo deploy/install.sh              install, or update to this checkout
#   sudo deploy/install.sh --port22     also stop port 22 from trusting ~/.ssh/authorized_keys
#   sudo deploy/install.sh --disable    stop everything; the windows go back to running alone
#
# What it sets up:
#   cutieplus account      a system account without a login, for the service and its database
#   /opt/cutieplus         a root-owned copy of the cooperation code and its own Python
#                          environment: nothing the desktop account can change runs as root or as
#                          the service
#   /etc/cutieplus         service.toml, client.toml, the annotation sshd's configuration, and the
#                          SSH policy file (ssh/authorized_keys, written by the helper only)
#   /var/lib/cutieplus     the database and its backups (the service's only)
#   systemd units          cutieplus-service, cutieplus-helper (root), cutieplus-gpu-worker,
#                          cutieplus-desktop@N, cutieplus-sshd (port 2222)
# The account's own sshd (port 22) is left alone unless --port22 is given.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DESKTOP_USER="${SUDO_USER:-}"
SSH_PORT=2222
APSW=apsw==3.53.4.0
ARGON2=argon2-cffi==25.1.0

say() { printf '\n== %s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die 'run it with sudo'
[ -n "$DESKTOP_USER" ] && [ "$DESKTOP_USER" != root ] || die 'run it with sudo from the desktop account'
DESKTOP_HOME="$(getent passwd "$DESKTOP_USER" | cut -d: -f6)"
UNITS=(cutieplus-service cutieplus-helper cutieplus-gpu-worker cutieplus-sshd)

if [ "${1:-}" = --disable ]; then
    say 'Stopping the cooperation module'
    systemctl stop 'cutieplus-desktop@*.service' 2>/dev/null || true
    for unit in "${UNITS[@]}"; do
        systemctl disable --now "$unit.service" 2>/dev/null || true
    done
    rm -f /etc/cutieplus/client.toml
    say 'Done: windows started from now on run the models themselves, as before.'
    say 'The database, keys and settings are kept in /var/lib/cutieplus and /etc/cutieplus.'
    exit 0
fi

if [ "${1:-}" = --port22 ]; then
    say 'Port 22: keys from a root-owned file only'
    install -d -m 0755 -o root -g root /etc/ssh/authorized_keys.d
    keys="/etc/ssh/authorized_keys.d/$DESKTOP_USER"
    if [ ! -e "$keys" ]; then
        if [ -s "$DESKTOP_HOME/.ssh/authorized_keys" ]; then
            install -m 0644 -o root -g root "$DESKTOP_HOME/.ssh/authorized_keys" "$keys"
        else
            install -m 0644 -o root -g root /dev/null "$keys"
        fi
    fi
    printf 'AuthorizedKeysFile /etc/ssh/authorized_keys.d/%%u\n' \
        > /etc/ssh/sshd_config.d/10-cutieplus-keys.conf
    sshd -t || { rm -f /etc/ssh/sshd_config.d/10-cutieplus-keys.conf; die 'sshd refused it; nothing changed'; }
    systemctl reload ssh.service 2>/dev/null || systemctl reload sshd.service 2>/dev/null || true
    say "Done: port 22 reads keys from $keys (root-owned). Password sign-in is unchanged."
    exit 0
fi

say 'Checking what is needed'
for program in /usr/bin/Xtigervnc /usr/bin/tigervncpasswd /usr/bin/openbox /usr/bin/dbus-run-session \
               /usr/sbin/sshd /usr/bin/python3; do
    [ -x "$program" ] || die "$program is missing (apt install tigervnc-standalone-server tigervnc-tools openbox dbus-x11 openssh-server)"
done
[ -x "$REPO/.venv/bin/python" ] || die "$REPO/.venv is missing: set the project up first (uv sync)"
UV="$(command -v uv || true)"
[ -n "$UV" ] || UV="$DESKTOP_HOME/.local/bin/uv"
[ -x "$UV" ] || die 'uv is needed to build the service environment'

say 'The cutieplus account'
if ! getent passwd cutieplus >/dev/null; then
    useradd --system --home-dir /var/lib/cutieplus --no-create-home --shell /usr/sbin/nologin cutieplus
fi

say 'The protected release in /opt/cutieplus'
release="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
target="/opt/cutieplus/releases/$release"
install -d -m 0755 -o root -g root /opt/cutieplus /opt/cutieplus/releases
rm -rf "$target.new"
install -d -m 0755 -o root -g root "$target.new/cutieplus" "$target.new/deploy"
cp -r "$REPO/cutieplus/__init__.py" "$REPO/cutieplus/cooperation" "$REPO/cutieplus/engine" \
    "$target.new/cutieplus/"
cp "$REPO/deploy/desktop-session" "$target.new/deploy/"
find "$target.new" -name __pycache__ -prune -exec rm -rf {} +
chown -R root:root "$target.new"
chmod -R u=rwX,go=rX "$target.new"
chmod 0755 "$target.new/deploy/desktop-session"
rm -rf "$target"
mv "$target.new" "$target"
ln -sfn "$target" /opt/cutieplus/current

say 'The service environment (/opt/cutieplus/venv)'
if [ ! -x /opt/cutieplus/venv/bin/python ]; then
    "$UV" venv --python /usr/bin/python3 /opt/cutieplus/venv
fi
numpy="numpy==$("$REPO/.venv/bin/python" -c 'import numpy; print(numpy.__version__)')"
"$UV" pip install --python /opt/cutieplus/venv/bin/python --quiet "$APSW" "$ARGON2" "$numpy"
site="$(/opt/cutieplus/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo /opt/cutieplus/current > "$site/cutieplus-release.pth"
chown -R root:root /opt/cutieplus/venv
chmod -R go-w /opt/cutieplus/venv

say 'Folders and the GPU lock'
install -d -m 0755 -o root -g root /etc/cutieplus /etc/cutieplus/ssh /var/lib/cutieplus-desktops \
    /var/lib/cutieplus-gpu
install -d -m 0700 -o cutieplus -g cutieplus /var/lib/cutieplus
[ -e /etc/cutieplus/ssh/authorized_keys ] || install -m 0644 -o root -g root /dev/null /etc/cutieplus/ssh/authorized_keys
[ -e /var/lib/cutieplus-gpu/gpu.lock ] || install -m 0644 -o root -g root /dev/null /var/lib/cutieplus-gpu/gpu.lock
sudo -u "$DESKTOP_USER" mkdir -p "$REPO/workspace/.cutie_jobs"

say 'Configuration (/etc/cutieplus)'
if [ ! -e /etc/cutieplus/service.toml ]; then
    cat > /etc/cutieplus/service.toml <<TOML
# The CutiePlus cooperation module (docs/COOPERATION.md). Changed by hand only; restart
# cutieplus-service, cutieplus-helper and cutieplus-gpu-worker afterwards.
[paths]
workspace_root = "$REPO/workspace"
tasks_dir = "$REPO/workspace/cutie_tasks"

[desktops]
first_display = 2
last_display = 20

[ssh]
ssh_port = $SSH_PORT
# how collaborators name this computer in their ssh command (a name or address they can reach)
ssh_host = "$(hostname -f 2>/dev/null || hostname)"
desktop_user = "$DESKTOP_USER"
TOML
fi
# paths and ports only (no secret): the GPU worker and the windows read it too
chown root:root /etc/cutieplus/service.toml
chmod 0644 /etc/cutieplus/service.toml
printf 'mode = "cooperation"\napp_socket = "/run/cutieplus/app.sock"\n' > /etc/cutieplus/client.toml
chmod 0644 /etc/cutieplus/client.toml

say "The annotation SSH entrypoint (port $SSH_PORT)"
/usr/bin/python3 -I -c "
import sys; sys.path.insert(0, '/opt/cutieplus/current')
from cutieplus.cooperation.config import ServiceConfig
from cutieplus.cooperation.sshpolicy import render_sshd_config
c = ServiceConfig.load('/etc/cutieplus/service.toml')
sys.stdout.write(render_sshd_config(c.ssh_port, c.authorized_keys, c.vnc_ports, user=c.desktop_user))
" > /etc/cutieplus/sshd_config.new
mkdir -p -m 0755 /run/sshd
sshd -t -f /etc/cutieplus/sshd_config.new || die 'sshd refused the annotation configuration'
mv /etc/cutieplus/sshd_config.new /etc/cutieplus/sshd_config
chmod 0644 /etc/cutieplus/sshd_config
if command -v ufw >/dev/null && ufw status | grep -q 'Status: active'; then
    ufw allow "$SSH_PORT/tcp" comment 'CutiePlus annotation SSH' >/dev/null
fi

say 'systemd units'
for unit in "$REPO"/deploy/units/*.service; do
    sed -e "s|@REPO@|$REPO|g" -e "s|@USER@|$DESKTOP_USER|g" -e "s|@HOME@|$DESKTOP_HOME|g" \
        "$unit" > "/etc/systemd/system/$(basename "$unit")"
done
cat > /usr/local/sbin/cutieplus-admin <<'SH'
#!/bin/sh
exec /opt/cutieplus/venv/bin/python -I -m cutieplus.cooperation.admin_cli "$@"
SH
chmod 0755 /usr/local/sbin/cutieplus-admin
systemctl daemon-reload
for unit in "${UNITS[@]}"; do
    systemctl enable "$unit.service" >/dev/null
    systemctl restart "$unit.service"
done

say 'Installed'
systemctl --no-pager --lines=0 status "${UNITS[@]/%/.service}" | grep -E '●|Active:' || true
cat <<TEXT

Next:
  1. sudo cutieplus-admin bootstrap-owner --login <your login> --name "<your name>"
     It prints a setup token; start Cutie at the monitor, choose "First time?" and set your
     password.
  2. Add collaborators in Manage Annotators (task browser).
  3. If they connect from outside this network, forward port $SSH_PORT to this computer.
  4. Once all works: sudo deploy/install.sh --port22 (port 22 stops trusting
     ~/.ssh/authorized_keys, which any window of the account could change).
TEXT
