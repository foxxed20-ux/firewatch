"""Checks the actual archive verifier against tampering and unsafe members."""

from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "deploy" / "verify_release.py"


def invoke(*args):
    return subprocess.run(
        [sys.executable, str(VERIFIER), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_bootstrap_package_is_immutable_and_detects_modified_code(tmp_path):
    source = tmp_path / "source"
    (source / "firewatch_service").mkdir(parents=True)
    (source / "firewatch_service" / "api.py").write_text("VERSION = 1\n")
    command = [
        sys.executable,
        str(ROOT / "deploy" / "package_release.py"),
        "--root",
        str(source),
        "--name",
        "test-r1",
        "--bootstrap-without-model",
    ]
    built = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert built.returncode == 0, built.stderr
    receipt = json.loads(built.stdout)
    target = tmp_path / "installed"
    verified = invoke(
        "--archive",
        receipt["archive"],
        "--sha256",
        receipt["sha256"],
        "--directory",
        target,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["status"] == "verified"
    duplicate = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert duplicate.returncode != 0
    (target / "firewatch_service" / "api.py").write_text("VERSION = 2\n")
    assert invoke("--directory", target).returncode != 0


@pytest.mark.parametrize(
    "name,link",
    [
        ("../outside.txt", False),
        ("a/../outside.txt", False),
        ("./duplicate.txt", False),
        ("link", True),
    ],
)
def test_pinned_archive_still_rejects_unsafe_members(tmp_path, name, link):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            tar.addfile(member)
        else:
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
    result = invoke(
        "--archive",
        archive,
        "--sha256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "--directory",
        tmp_path / "target",
    )
    assert result.returncode != 0
    assert not (tmp_path / "target").exists()
    assert not (tmp_path / "outside.txt").exists()


def test_activation_probe_supports_bearer_without_secret_in_argv(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.headers.get("Authorization"))
            status = 200 if observed[-1] == "Bearer fixture-token" else 401
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"status":"ready"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = tmp_path / "service.env"
    env.write_text('FIREWATCH_PUBLIC_DEMO=false\nFIREWATCH_TOKEN="fixture-token"\n')
    try:
        args = [
            sys.executable,
            str(ROOT / "deploy" / "check_readiness.py"),
            "--env-file",
            str(env),
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
        ]
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert observed == ["Bearer fixture-token"]
        assert "fixture-token" not in result.stdout + result.stderr + " ".join(args)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
