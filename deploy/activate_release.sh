#!/usr/bin/env bash
# Run as red. A release directory must already contain a checked service candidate.
set -euo pipefail
base=/home/red/firewatch
release=${1:?release directory name required}
health=ready
if test "${2:-}" = "--bootstrap-without-model"; then health=live; fi
case "$release" in *[!A-Za-z0-9._-]*|.|..) echo 'unsafe release name' >&2; exit 2;; esac
target="$base/releases/$release"
test -f "$target/firewatch_service/api.py"
test -f "$target/web/index.html"
"$base/venv/bin/python" "$base/deploy/verify_release.py" --directory "$target"
"$base/venv/bin/python" -m compileall -q "$target/firewatch_service"
previous=$(readlink "$base/current" || true)
if test -n "$previous"; then printf '%s\n' "$previous" > "$base/shared/previous_release"; fi
ln -sfn "$target" "$base/current.next"
mv -Tf "$base/current.next" "$base/current"
sudo -n systemctl restart firewatch
for attempt in $(seq 1 20); do
    if "$base/venv/bin/python" "$base/deploy/check_readiness.py" --env-file "$base/shared/service.env" --path "$health"; then
        echo "activated $release"
        exit 0
    fi
    sleep 1
done
if test -n "$previous"; then
    ln -sfn "$previous" "$base/current.next"
    mv -Tf "$base/current.next" "$base/current"
    sudo -n systemctl restart firewatch
fi
echo 'activation smoke failed; previous release restored when available' >&2
exit 1
