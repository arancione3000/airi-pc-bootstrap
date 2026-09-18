from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(os.environ.get("AIRI_ROOT") or os.environ.get("AIRIPC_WORKSPACE_ROOT") or Path(__file__).resolve().parents[2]).resolve()
CONFIG_PATH = ROOT / ".ai" / "airi_control.json"
PAIR_PATH = ROOT / ".ai" / "state" / "phone_pair_secret"


def _b64u_decode(value: str) -> bytes:
    value = value.strip()
    value += "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value.encode())


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def pair_secret(value: str | None = None) -> bytes:
    raw = value or os.environ.get("AIRI_PHONE_PAIR_SECRET", "").strip()
    if not raw:
        try:
            raw = PAIR_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            raw = ""
    if not raw:
        raise RuntimeError("Airi Control is not paired. Run: python -m computer.control_plane.phone_control pair <code>")
    decoded = _b64u_decode(raw)
    if len(decoded) != 32:
        raise ValueError("pairing code must decode to exactly 32 bytes")
    return decoded


def save_pair_secret(value: str) -> dict[str, Any]:
    decoded = _b64u_decode(value)
    if len(decoded) != 32:
        raise ValueError("pairing code must decode to exactly 32 bytes")
    PAIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAIR_PATH.write_text(value.strip() + "\n", encoding="utf-8")
    PAIR_PATH.chmod(0o600)
    return {"ok": True, "auth_id": hashlib.sha256(decoded).digest()[:8].hex(), "path": str(PAIR_PATH)}


def _config() -> dict[str, str]:
    data = {
        "relay_base": "https://ntfy.sh",
        "topic": "airi-control-6f4ea987afbe4cc9b71c11f0421e4d88555ee508c97a4102",
        "history": "30m",
    }
    try:
        row = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(row, dict):
            for key in data:
                if row.get(key):
                    data[key] = str(row[key])
    except Exception:
        pass
    env_relay = os.environ.get("AIRI_CONTROL_RELAY_BASE", "").strip()
    env_topic = os.environ.get("AIRI_CONTROL_TOPIC", "").strip()
    if env_relay:
        data["relay_base"] = env_relay
    if env_topic:
        data["topic"] = env_topic
    return data


def _auth_id(secret: bytes) -> str:
    return hashlib.sha256(secret).digest()[:8].hex()


def _tag(secret: bytes, *fields: str) -> str:
    raw = "\n".join(fields).encode()
    return base64.urlsafe_b64encode(hmac.new(secret, raw, hashlib.sha256).digest()).decode().rstrip("=")


def _publish(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = _config()
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"{cfg['relay_base'].rstrip('/')}/{cfg['topic']}",
        data=body,
        method="POST",
        headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "Airi-PC-Phone-Control/0.1"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        text = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _messages(since: str = "30m") -> list[dict[str, Any]]:
    cfg = _config()
    url = f"{cfg['relay_base'].rstrip('/')}/{cfg['topic']}/json?poll=1&since={since}"
    req = urllib.request.Request(url, headers={"User-Agent": "Airi-PC-Phone-Control/0.1"})
    with urllib.request.urlopen(req, timeout=20) as response:
        lines = response.read().decode("utf-8", errors="replace").splitlines()
    out = []
    for line in lines:
        try:
            outer = json.loads(line)
            if outer.get("event") != "message":
                continue
            packet = json.loads(outer.get("message") or "{}")
            packet["_relay_time"] = int(outer.get("time") or 0)
            out.append(packet)
        except Exception:
            continue
    return out


def _discover_device(secret: bytes, timeout: float = 30.0) -> dict[str, Any]:
    aid = _auth_id(secret)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = [
            m for m in _messages("30m")
            if m.get("kind") == "phone_device_key"
            and m.get("auth_id") == aid
            and m.get("device_key_id")
            and m.get("public_key_b64")
        ]
        if candidates:
            return max(candidates, key=lambda x: int(x.get("ts") or x.get("_relay_time") or 0))
        time.sleep(1.0)
    raise TimeoutError("paired phone not found on the control relay")


def _controller_keypair() -> tuple[Any, str, str]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_der = private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_b64 = _b64(public_der)
    key_id = hashlib.sha256(public_der).digest()[:12].hex()
    return private, key_id, public_b64


def _hybrid_encrypt(public_b64: str, plain: bytes) -> dict[str, str]:
    public = serialization.load_der_public_key(base64.b64decode(public_b64))
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, plain, None)
    wrapped = public.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "wrapped_key_b64": _b64(wrapped),
        "nonce_b64": _b64(nonce),
        "ciphertext_b64": _b64(ciphertext),
    }


def _hybrid_decrypt(private: Any, packet: dict[str, Any]) -> bytes:
    key = private.decrypt(
        base64.b64decode(packet["wrapped_key_b64"]),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return AESGCM(key).decrypt(
        base64.b64decode(packet["nonce_b64"]),
        base64.b64decode(packet["ciphertext_b64"]),
        None,
    )


def _verify_result_tag(secret: bytes, packet: dict[str, Any]) -> bool:
    expected = _tag(
        secret,
        "phone_result",
        str(packet.get("device_key_id") or ""),
        str(packet.get("controller_key_id") or ""),
        str(packet.get("command_id") or ""),
        str(packet.get("wrapped_key_b64") or ""),
        str(packet.get("nonce_b64") or ""),
        str(packet.get("ciphertext_b64") or ""),
        str(packet.get("ts") or ""),
        str(packet.get("auth_id") or ""),
    )
    return hmac.compare_digest(expected, str(packet.get("auth_tag") or ""))


def send(op: str, args: dict[str, Any] | None = None, *, timeout: float = 45.0, secret_text: str | None = None) -> dict[str, Any]:
    secret = pair_secret(secret_text)
    device = _discover_device(secret, timeout=min(timeout, 30.0))
    private, controller_id, controller_public = _controller_keypair()
    now = int(time.time())
    aid = _auth_id(secret)
    controller = {
        "kind": "phone_controller_key",
        "controller_key_id": controller_id,
        "public_key_b64": controller_public,
        "auth_id": aid,
        "ts": now,
        "protocol": 1,
    }
    controller["auth_tag"] = _tag(
        secret,
        "phone_controller_key",
        controller_id,
        controller_public,
        str(now),
        aid,
    )
    _publish(controller)
    time.sleep(0.7)

    command_id = uuid.uuid4().hex[:20]
    inner = {
        "id": command_id,
        "op": op,
        "args": args or {},
        "issued_at": int(time.time()),
        "expires_at": int(time.time()) + min(120, max(20, int(timeout) + 10)),
    }
    envelope = _hybrid_encrypt(device["public_key_b64"], json.dumps(inner, separators=(",", ":")).encode())
    ts = int(time.time())
    packet = {
        "kind": "phone_cmd",
        "device_key_id": device["device_key_id"],
        "controller_key_id": controller_id,
        "auth_id": aid,
        "ts": ts,
        "protocol": 1,
        **envelope,
    }
    packet["auth_tag"] = _tag(
        secret,
        "phone_cmd",
        packet["device_key_id"],
        controller_id,
        packet["wrapped_key_b64"],
        packet["nonce_b64"],
        packet["ciphertext_b64"],
        str(ts),
        aid,
    )
    _publish(packet)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for result in reversed(_messages("10m")):
            if (
                result.get("kind") != "phone_result"
                or result.get("command_id") != command_id
                or result.get("controller_key_id") != controller_id
                or result.get("auth_id") != aid
                or not _verify_result_tag(secret, result)
            ):
                continue
            plain = json.loads(_hybrid_decrypt(private, result).decode("utf-8"))
            return plain
        time.sleep(0.8)
    raise TimeoutError(f"phone command timed out: {op}")


def screenshot(path: str, *, timeout: float = 60.0, secret_text: str | None = None) -> dict[str, Any]:
    result = send("screenshot", timeout=timeout, secret_text=secret_text)
    if not result.get("ok"):
        return result
    url = result.get("frame_url")
    if not url:
        raise RuntimeError("phone returned no frame URL")
    req = urllib.request.Request(str(url), headers={"User-Agent": "Airi-PC-Phone-Control/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        encrypted = response.read()
    key = base64.b64decode(result["frame_key_b64"])
    nonce = base64.b64decode(result["frame_nonce_b64"])
    jpeg = AESGCM(key).decrypt(nonce, encrypted, None)
    dest = Path(path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(jpeg)
    result["path"] = str(dest)
    result["bytes"] = len(jpeg)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airi-phone-control")
    parser.add_argument("--secret", default=None)
    parser.add_argument("--timeout", type=float, default=45.0)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pair = sub.add_parser("pair")
    p_pair.add_argument("code")

    sub.add_parser("status")
    sub.add_parser("observe")
    sub.add_parser("back")
    sub.add_parser("home")
    sub.add_parser("recents")
    sub.add_parser("stop")

    p_tap = sub.add_parser("tap")
    p_tap.add_argument("x", type=float)
    p_tap.add_argument("y", type=float)

    p_swipe = sub.add_parser("swipe")
    p_swipe.add_argument("x1", type=float)
    p_swipe.add_argument("y1", type=float)
    p_swipe.add_argument("x2", type=float)
    p_swipe.add_argument("y2", type=float)
    p_swipe.add_argument("--duration-ms", type=int, default=450)

    p_text = sub.add_parser("text")
    p_text.add_argument("value")

    p_shot = sub.add_parser("screenshot")
    p_shot.add_argument("path", nargs="?", default="/tmp/airi-phone.jpg")

    ns = parser.parse_args(argv)
    try:
        if ns.command == "pair":
            result = save_pair_secret(ns.code)
        elif ns.command == "tap":
            result = send("tap", {"x": ns.x, "y": ns.y}, timeout=ns.timeout, secret_text=ns.secret)
        elif ns.command == "swipe":
            result = send(
                "swipe",
                {"x1": ns.x1, "y1": ns.y1, "x2": ns.x2, "y2": ns.y2, "duration_ms": ns.duration_ms},
                timeout=ns.timeout,
                secret_text=ns.secret,
            )
        elif ns.command == "text":
            result = send("text", {"text": ns.value}, timeout=ns.timeout, secret_text=ns.secret)
        elif ns.command == "screenshot":
            result = screenshot(ns.path, timeout=ns.timeout, secret_text=ns.secret)
        else:
            result = send(ns.command, timeout=ns.timeout, secret_text=ns.secret)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok", True) else 2
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
