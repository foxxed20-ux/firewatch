"""Probe public or Bearer-protected service without placing credentials in argv."""

import argparse
import json
from pathlib import Path
import shlex
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8089")
    parser.add_argument("--path", choices=["live", "ready"], default="ready")
    args = parser.parse_args()
    settings = {}
    for line in Path(args.env_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator:
            parts = shlex.split(value, comments=False)
            settings[key.strip()] = " ".join(parts)
    headers = {}
    public = settings.get("FIREWATCH_PUBLIC_DEMO", "").lower() in {"1", "true", "yes"}
    if not public and settings.get("FIREWATCH_TOKEN"):
        headers["Authorization"] = "Bearer " + settings["FIREWATCH_TOKEN"]
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/health/" + args.path, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            data = json.load(response)
            expected = "ready" if args.path == "ready" else "ok"
            return 0 if response.status == 200 and data.get("status") == expected else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
