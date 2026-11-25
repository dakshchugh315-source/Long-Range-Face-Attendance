#!/usr/bin/env bash
# Deploy the online dashboard to an Ubuntu server that already runs Caddy.
#
#   bash cloud/deploy.sh <ssh-key.pem> <user@host> <domain>
#   e.g. bash cloud/deploy.sh ~/LightsailKey.pem ubuntu@1.2.3.4 attendance.1.2.3.4.sslip.io
#
# Safe on a server that hosts other sites: own folder (/opt/attendance-cloud), own system
# user, own service on 127.0.0.1:8090, and ONE extra Caddy site block. The Caddyfile is
# backed up and validated before reload; if validation fails the backup is restored.
# First run prints the teacher password and the sync token ONCE.
set -euo pipefail
KEY="$1"; HOST="$2"; DOMAIN="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")

echo "==> copying files"
"${SSH[@]}" "sudo mkdir -p /opt/attendance-cloud && sudo chown \$USER /opt/attendance-cloud"
scp -i "$KEY" "$HERE/cloud_app.py" "$HERE/teacher.html" "$HERE/student.html" "$HERE/requirements.txt" \
    "$HOST:/opt/attendance-cloud/"

echo "==> installing (venv, service, caddy site)"
"${SSH[@]}" DOMAIN="$DOMAIN" bash -s <<'REMOTE'
set -euo pipefail
cd /opt/attendance-cloud
sudo apt-get install -y -qq python3-venv >/dev/null
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
id attendance >/dev/null 2>&1 || sudo useradd --system --no-create-home --shell /usr/sbin/nologin attendance
sudo mkdir -p /var/lib/attendance-cloud && sudo chown attendance /var/lib/attendance-cloud && sudo chmod 700 /var/lib/attendance-cloud

if [ ! -f /etc/attendance-cloud.env ]; then
  PW=$(python3 -c "import secrets; print(secrets.token_urlsafe(9))")
  TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
  KEYS=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  printf 'SYNC_TOKEN=%s\nTEACHER_PASSWORD=%s\nSECRET_KEY=%s\nDATA_DIR=/var/lib/attendance-cloud\n' "$TOKEN" "$PW" "$KEYS" \
    | sudo tee /etc/attendance-cloud.env >/dev/null
  sudo chmod 600 /etc/attendance-cloud.env
  echo "######## SAVE THESE NOW (shown only once) ########"
  echo "Teacher password : $PW"
  echo "Sync token       : $TOKEN"
  echo "##################################################"
fi

sudo tee /etc/systemd/system/attendance-cloud.service >/dev/null <<UNIT
[Unit]
Description=Face Attendance online dashboard
After=network.target
[Service]
User=attendance
WorkingDirectory=/opt/attendance-cloud
EnvironmentFile=/etc/attendance-cloud.env
ExecStart=/opt/attendance-cloud/.venv/bin/gunicorn -w 2 -b 127.0.0.1:8090 cloud_app:app
Restart=always
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now attendance-cloud >/dev/null
sudo systemctl restart attendance-cloud

if ! sudo grep -q "^$DOMAIN {" /etc/caddy/Caddyfile; then
  sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak.$(date +%s)
  printf '\n%s {\n\treverse_proxy 127.0.0.1:8090\n}\n' "$DOMAIN" | sudo tee -a /etc/caddy/Caddyfile >/dev/null
  if sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1; then
    sudo systemctl reload caddy
  else
    echo "Caddy config invalid - restoring backup, other sites untouched"
    sudo cp "$(ls -t /etc/caddy/Caddyfile.bak.* | head -1)" /etc/caddy/Caddyfile
    exit 1
  fi
fi
sleep 2
curl -fsS http://127.0.0.1:8090/healthz && echo
REMOTE
echo "==> done: https://$DOMAIN  (teacher)   https://$DOMAIN/student  (students)"
echo "    In the laptop app: Students tab -> Online dashboard -> address https://$DOMAIN + the sync token"
