from __future__ import annotations

import atexit
import base64
import hmac
import io
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw

from .live_telemetry import DEFAULT_RELAY, live_emit, session_id, topic_for

ROOT = Path(os.environ.get("AIRI_ROOT") or os.environ.get("AIRIPC_WORKSPACE_ROOT") or ".").resolve()
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def _relay_base() -> str:
    return os.environ.get("AIRI_LIVE_RELAY", DEFAULT_RELAY).rstrip("/")


def _post_json(payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{_relay_base()}/{topic_for()}",
                data=body,
                method="POST",
                headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "Airi-PC-Screen/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"relay publish failed: HTTP {response.status}")
                response.read(1)
            return
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(str(last or "relay publish failed"))


def _viewer_keys(since: str = "2h") -> list[dict]:
    url = f"{_relay_base()}/{topic_for()}/json?poll=1&since={since}"
    req = urllib.request.Request(url, headers={"User-Agent": "Airi-PC-Screen/1.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        lines = response.read().decode("utf-8", errors="replace").splitlines()
    latest = {}
    now = int(time.time())
    for line in lines:
        try:
            outer = json.loads(line)
            inner = json.loads(outer.get("message") or "{}")
        except Exception:
            continue
        if inner.get("kind") != "viewer_key":
            continue
        key_id = str(inner.get("key_id") or "").strip()
        public_key_b64 = str(inner.get("public_key_b64") or "").strip()
        ts = int(inner.get("ts") or 0)
        if not key_id or not public_key_b64 or ts <= 0 or now - ts > 3 * 3600:
            continue
        row = {"key_id": key_id, "public_key_b64": public_key_b64, "ts": ts}
        previous = latest.get(key_id)
        if previous is None or ts >= previous["ts"]:
            latest[key_id] = row
    return sorted(latest.values(), key=lambda x: x["ts"], reverse=True)[:5]


def _encrypt_descriptor(public_key_b64: str, descriptor: dict) -> str:
    openssl = shutil.which("openssl")
    if not openssl:
        raise RuntimeError("openssl is required for encrypted screen offers")
    payload = json.dumps(descriptor, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > 280:
        raise ValueError("screen descriptor too large for RSA-OAEP envelope")
    public_der = base64.b64decode(public_key_b64, validate=True)
    with tempfile.TemporaryDirectory(prefix="airi-screen-key-") as td:
        base = Path(td)
        der = base / "public.der"
        pem = base / "public.pem"
        plain = base / "plain.bin"
        cipher = base / "cipher.bin"
        der.write_bytes(public_der)
        plain.write_bytes(payload)
        subprocess.run(
            [openssl, "pkey", "-pubin", "-inform", "DER", "-in", str(der), "-out", str(pem)],
            check=True, capture_output=True, timeout=10,
        )
        subprocess.run(
            [
                openssl, "pkeyutl", "-encrypt", "-pubin", "-inkey", str(pem),
                "-in", str(plain), "-out", str(cipher),
                "-pkeyopt", "rsa_padding_mode:oaep",
                "-pkeyopt", "rsa_oaep_md:sha256",
                "-pkeyopt", "rsa_mgf1_md:sha1",
            ],
            check=True, capture_output=True, timeout=10,
        )
        return base64.b64encode(cipher.read_bytes()).decode("ascii")


def _download_cloudflared() -> Path:
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)
    dest = ROOT / ".ai" / "bin" / "cloudflared"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and os.access(dest, os.X_OK):
        return dest
    tmp = dest.with_suffix(".download")
    req = urllib.request.Request(CLOUDFLARED_URL, headers={"User-Agent": "Airi-PC-Screen/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.chmod(0o700)
    tmp.replace(dest)
    return dest


def _ensure_display() -> None:
    if os.environ.get("DISPLAY"):
        return
    xvfb = shutil.which("Xvfb")
    if not xvfb:
        raise RuntimeError("No graphical DISPLAY and Xvfb is unavailable")
    display = ":99"
    proc = subprocess.Popen(
        [xvfb, display, "-screen", "0", "1280x800x24", "-ac"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    os.environ["DISPLAY"] = display
    os.environ.setdefault("AIRI_BROWSER_HEADLESS", "0")
    time.sleep(0.8)
    if proc.poll() is not None:
        raise RuntimeError("Xvfb failed to start")


def _capture_jpeg() -> bytes:
    from server import mouse_position, screenshot_image

    img = screenshot_image().convert("RGB")
    try:
        pos = mouse_position()
        draw = ImageDraw.Draw(img)
        x, y = int(pos["x"]), int(pos["y"])
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(255, 138, 0), width=4)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 138, 0))
    except Exception:
        pass
    if img.width > 1100:
        scale = 1100 / img.width
        img = img.resize((1100, int(img.height * scale)), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=62, optimize=True)
    return out.getvalue()


class _ViewerHandler(BaseHTTPRequestHandler):
    server_version = "AiriScreen/1.1"

    def _authorized(self, token: str) -> bool:
        expected = getattr(self.server, "airi_token", "")
        return bool(expected and hmac.compare_digest(token, expected))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
        if len(parts) != 2 or parts[0] not in {"view", "frame", "stream", "health"}:
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        kind, token_part = parts
        token = token_part[:-4] if kind == "frame" and token_part.endswith(".jpg") else token_part
        if not self._authorized(token):
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        if kind == "health":
            self._send(200, "application/json", b'{"ok":true}')
            return
        if kind == "frame":
            try:
                body = _capture_jpeg()
            except Exception as exc:
                self._send(503, "text/plain; charset=utf-8", str(exc).encode("utf-8", errors="replace")[:1000])
                return
            self._send(200, "image/jpeg", body)
            return
        if kind == "stream":
            boundary = b"airiframe"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=airiframe")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            target_interval = 0.10
            try:
                while True:
                    started = time.monotonic()
                    body = _capture_jpeg()
                    self.wfile.write(b"--" + boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(body)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    remaining = target_interval - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception:
                pass
            return
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Airi-PC POV</title><style>
html,body{{margin:0;width:100%;height:100%;background:#050507;overflow:hidden}}
body{{display:flex;align-items:center;justify-content:center}}
#screen{{width:100%;height:100%;object-fit:contain;background:#050507}}
#badge{{position:fixed;top:8px;left:8px;padding:5px 8px;border-radius:10px;background:#000a;color:#5ed390;
font:600 12px system-ui,sans-serif;letter-spacing:.04em}}
</style></head><body><img id="screen" src="/stream/{token}" alt="Airi-PC live screen"><div id="badge">AIRI-PC · LIVE POV · MJPEG</div>
<script>
const img=document.getElementById('screen');
img.onerror=()=>setTimeout(()=>{img.src='/stream/{token}?t='+Date.now()},1200);
</script></body></html>"""
        self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))

    def log_message(self, fmt: str, *args) -> None:
        return


class ScreenShare:
    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(32)
        self.httpd = None
        self.server_thread = None
        self.tunnel = None
        self.tunnel_url = ""
        self.offer_thread = None
        self.stop_event = threading.Event()
        self.offered = set()

    def _start_http(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ViewerHandler)
        setattr(self.httpd, "airi_token", self.token)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, name="airi-screen-http", daemon=True)
        self.server_thread.start()

    @property
    def port(self) -> int:
        if self.httpd is None:
            raise RuntimeError("screen HTTP server is not running")
        return int(self.httpd.server_address[1])

    def _start_tunnel(self) -> None:
        cloudflared = _download_cloudflared()
        self.tunnel = subprocess.Popen(
            [str(cloudflared), "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate", "--protocol", "http2"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        q = queue.Queue()

        def reader() -> None:
            if self.tunnel is None or self.tunnel.stdout is None:
                return
            for line in self.tunnel.stdout:
                q.put(line)

        threading.Thread(target=reader, name="airi-cloudflared-log", daemon=True).start()
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            if self.tunnel.poll() is not None:
                raise RuntimeError("cloudflared exited before creating a quick tunnel")
            try:
                line = q.get(timeout=0.5)
            except queue.Empty:
                continue
            match = _TUNNEL_RE.search(line)
            if match:
                self.tunnel_url = match.group(0).rstrip("/")
                return
        raise TimeoutError("cloudflared quick tunnel URL was not created in time")

    def _external_self_test(self) -> None:
        health = f"{self.tunnel_url}/health/{self.token}"
        frame = f"{self.tunnel_url}/frame/{self.token}.jpg"
        last = None
        for _ in range(15):
            try:
                with urllib.request.urlopen(health, timeout=12) as response:
                    if response.status != 200:
                        raise RuntimeError(f"health HTTP {response.status}")
                    response.read()
                with urllib.request.urlopen(frame, timeout=20) as response:
                    data = response.read(1_500_000)
                    ctype = response.headers.get("content-type", "")
                    if response.status != 200 or "image/jpeg" not in ctype or len(data) < 2000:
                        raise RuntimeError("public frame smoke test failed")
                return
            except Exception as exc:
                last = exc
                time.sleep(1.5)
        raise RuntimeError(f"screen tunnel external self-test failed: {type(last).__name__ if last else 'unknown'}")

    def _offer_loop(self) -> None:
        descriptor = {"u": f"{self.tunnel_url}/view/{self.token}", "s": session_id(), "v": 1}
        while not self.stop_event.wait(0.1):
            try:
                for row in _viewer_keys():
                    key_id = row["key_id"]
                    if key_id in self.offered:
                        continue
                    cipher = _encrypt_descriptor(row["public_key_b64"], descriptor)
                    _post_json({
                        "kind": "screen_offer",
                        "key_id": key_id,
                        "ciphertext_b64": cipher,
                        "session_id": session_id(),
                        "ts": int(time.time()),
                        "protocol": 3,
                    })
                    self.offered.add(key_id)
                if self.stop_event.wait(3.0):
                    break
            except Exception:
                if self.stop_event.wait(3.0):
                    break

    def start(self):
        _ensure_display()
        print("AIRI_POV_STAGE=display", flush=True)
        self._start_http()
        print("AIRI_POV_STAGE=http", flush=True)
        self._start_tunnel()
        print("AIRI_POV_STAGE=tunnel", flush=True)

        def verify_public_path() -> None:
            try:
                self._external_self_test()
                print("AIRI_POV_TUNNEL_SMOKE=PASS", flush=True)
            except Exception:
                print("AIRI_POV_TUNNEL_SMOKE=DEFERRED", flush=True)

        threading.Thread(target=verify_public_path, name="airi-screen-public-smoke", daemon=True).start()
        print("AIRI_POV_MJPEG_SERVER=READY", flush=True)
        self.offer_thread = threading.Thread(target=self._offer_loop, name="airi-screen-offers", daemon=True)
        self.offer_thread.start()
        live_emit("runtime", "POV screen ready", "Encrypted live viewer available for Airi Live", "completed", dedupe_key="screen-share:ready")
        return self

    def close(self) -> None:
        self.stop_event.set()
        try:
            if self.httpd is not None:
                self.httpd.shutdown()
                self.httpd.server_close()
        except Exception:
            pass
        try:
            if self.tunnel is not None and self.tunnel.poll() is None:
                self.tunnel.terminate()
                self.tunnel.wait(timeout=5)
        except Exception:
            try:
                if self.tunnel is not None:
                    self.tunnel.kill()
            except Exception:
                pass


_ACTIVE = None


def start_screen_share():
    global _ACTIVE
    enabled = os.environ.get("AIRI_SCREEN_SHARE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if not enabled or os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    share = ScreenShare()
    try:
        _ACTIVE = share.start()
        atexit.register(_ACTIVE.close)
        return _ACTIVE
    except Exception as exc:
        share.close()
        live_emit("runtime", "POV screen unavailable", str(exc)[:240], "failed", dedupe_key="screen-share:failed")
        return None
