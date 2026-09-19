#!/usr/bin/env bash
set -euo pipefail
base=/home/red/firewatch
install -m 0644 "$base/deploy/firewatch.service" /etc/systemd/system/firewatch.service
install -m 0644 "$base/deploy/firewatch-tunnel.service" /etc/systemd/system/firewatch-tunnel.service
printf '%s\n' 'red ALL=(root) NOPASSWD: /usr/bin/systemctl restart firewatch, /usr/bin/systemctl restart firewatch-tunnel' > /etc/sudoers.d/firewatch
chmod 0440 /etc/sudoers.d/firewatch
visudo -cf /etc/sudoers.d/firewatch
chmod 0600 "$base/deploy/tunnel_key"
chmod 0644 "$base/deploy/known_hosts"
chown red:red "$base/deploy/tunnel_key" "$base/deploy/known_hosts"
systemctl daemon-reload
systemctl enable firewatch firewatch-tunnel
systemctl restart firewatch-tunnel
systemctl is-active firewatch-tunnel
