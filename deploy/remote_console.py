"""Temporary SSH control session. Credentials arrive via stdin; never persisted.

Run with unbuffered Python. Every input/output is a JSON object on one line.
Remote mutation commands are supplied explicitly by the operator, never inferred.
"""

import json
import sys
from pathlib import Path

import paramiko

clients = {}

# A live PTY is needed to keep stdin available across tool calls. Disable input
# echo before any credentials arrive, so passwords cannot appear in PTY output.
if sys.stdin.isatty():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.restype = wintypes.HANDLE
        handle = kernel.GetStdHandle(-10)
        mode = wintypes.DWORD()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            raise RuntimeError("Cannot disable console echo")
        if not kernel.SetConsoleMode(handle, mode.value & ~0x0004):
            raise RuntimeError("Cannot disable console echo")
    else:
        import termios

        mode = termios.tcgetattr(sys.stdin.fileno())
        mode[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, mode)


def handle(item):
    action = item["action"]
    name = item.get("name", "app")
    if action == "connect":
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            item["host"],
            port=item.get("port", 22),
            username=item["user"],
            password=item["password"],
            timeout=15,
            auth_timeout=15,
            banner_timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
        transport = client.get_transport()
        transport.set_keepalive(25)
        clients[name] = client
        return {
            "connected": name,
            "host": item["host"],
            "host_key_sha256": __import__("base64")
            .b64encode(
                __import__("hashlib")
                .sha256(transport.get_remote_server_key().asbytes())
                .digest()
            )
            .decode(),
        }
    client = clients[name]
    if action == "exec":
        stdin, stdout, stderr = client.exec_command(
            item["command"], timeout=item.get("timeout", 60)
        )
        if item.get("stdin") is not None:
            stdin.write(item["stdin"])
            stdin.flush()
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return {
            "name": name,
            "exit_code": stdout.channel.recv_exit_status(),
            "stdout": out[-item.get("max_chars", 24000) :],
            "stderr": err[-4000:],
        }
    if action == "put":
        with client.open_sftp() as sftp:
            sftp.put(item["local"], item["remote"])
        return {"uploaded": item["remote"], "bytes": Path(item["local"]).stat().st_size}
    if action == "get":
        dest = Path(item["local"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            sftp.get(item["remote"], str(dest))
        return {"downloaded": str(dest), "bytes": dest.stat().st_size}
    raise ValueError("Unsupported action")


print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    try:
        item = json.loads(line)
        if item.get("action") == "close":
            break
        result = handle(item)
        if item.get("action") == "exec":
            log = Path("deploy/private") / (
                "last_ssh_" + item.get("name", "app") + ".json"
            )
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "name": result["name"],
                        "exit_code": result["exit_code"],
                        "output_file": str(log),
                    }
                ),
                flush=True,
            )
        else:
            print(
                json.dumps({"ok": True, "result": result}, ensure_ascii=False),
                flush=True,
            )
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            ),
            flush=True,
        )
for client in clients.values():
    client.close()
