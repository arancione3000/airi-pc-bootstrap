from __future__ import annotations

import argparse
import base64
import hashlib
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

ROOT = Path(
    os.environ.get("AIRI_ROOT")
    or os.environ.get("AIRIPC_WORKSPACE_ROOT")
    or Path(__file__).resolve().parents[2]
).resolve()
CONFIG_PATH = ROOT / ".ai" / "airi_control.json"
SESSION_PATH = ROOT / ".ai" / "state" / "phone_control_controller.json"

DEFAULT_RELAY = "https://ntfy.sh"
DEFAULT_BOOTSTRAP = "airi-control-bootstrap-2e6a6f6c97314f2ba8d6c41e2fa1e4d2"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _canonical(*fields: str) -> bytes:
    return "\n".join(fields).encode("utf-8")


def _key_id(public_der: bytes) -> str:
    return hashlib.sha256(public_der).digest()[:12].hex()


def _config() -> dict[str, str]:
    data = {
        "relay_base": DEFAULT_RELAY,
        "bootstrap_topic": DEFAULT_BOOTSTRAP,
    }
    try:
        row = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(row, dict):
            if row.get("relay_base"):
                data["relay_base"] = str(row["relay_base"])
            if row.get("bootstrap_topic"):
                data["bootstrap_topic"] = str(row["bootstrap_topic"])
            elif row.get("topic"):
                data["bootstrap_topic"] = str(row["topic"])
    except Exception:
        pass

    env_relay = os.environ.get("AIRI_CONTROL_RELAY_BASE", "").strip()
    env_topic = os.environ.get("AIRI_CONTROL_TOPIC", "").strip()
    if env_relay:
        data["relay_base"] = env_relay
    if env_topic:
        data["bootstrap_topic"] = env_topic
    return data


def _publish(topic: str, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = _config()
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"{cfg['relay_base'].rstrip('/')}/{topic}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "Airi-PC-Phone-Control/0.2",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        text = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def _messages(topic: str, since: str = "2m") -> list[dict[str, Any]]:
    cfg = _config()
    url = f"{cfg['relay_base'].rstrip('/')}/{topic}/json?poll=1&since={since}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Airi-PC-Phone-Control/0.2"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        lines = response.read().decode("utf-8", errors="replace").splitlines()

    out: list[dict[str, Any]] = []
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


def _generate_controller() -> tuple[Any, str, str]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_der = private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private, _key_id(public_der), _b64(public_der)


def _sign(private: Any, data: bytes) -> str:
    return _b64(
        private.sign(
            data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    )


def _verify(public_b64: str, data: bytes, signature_b64: str) -> bool:
    try:
        public = serialization.load_der_public_key(base64.b64decode(public_b64))
        public.verify(
            base64.b64decode(signature_b64),
            data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


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


def _verify_ready(row: dict[str, Any]) -> bool:
    try:
        signed = _canonical(
            "phone_ready",
            str(row["session_id"]),
            str(row["device_key_id"]),
            str(row["encryption_public_key_b64"]),
            str(row["signing_public_key_b64"]),
            str(row["session_topic"]),
            str(row["ts"]),
            str(row["expires_at"]),
        )
        return _verify(
            str(row["signing_public_key_b64"]),
            signed,
            str(row["device_signature_b64"]),
        )
    except Exception:
        return False


def _discover_ready(timeout: float = 30.0) -> dict[str, Any]:
    cfg = _config()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        now = int(time.time())
        rows = [
            row
            for row in _messages(cfg["bootstrap_topic"], "2m")
            if row.get("kind") == "phone_ready"
            and int(row.get("expires_at") or 0) >= now
            and row.get("session_id")
            and row.get("session_topic")
            and row.get("device_key_id")
            and row.get("encryption_public_key_b64")
            and row.get("signing_public_key_b64")
            and _verify_ready(row)
        ]
        if rows:
            return max(
                rows,
                key=lambda x: int(x.get("ts") or x.get("_relay_time") or 0),
            )
        time.sleep(0.8)
    raise TimeoutError(
        "Airi Control non trovato. Premi AVVIA CONTROLLO sul telefono."
    )


def _save_session(
    ready: dict[str, Any],
    private: Any,
    controller_id: str,
    controller_public_b64: str,
) -> dict[str, Any]:
    pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    row = {
        "session_id": str(ready["session_id"]),
        "session_topic": str(ready["session_topic"]),
        "device_key_id": str(ready["device_key_id"]),
        "device_encryption_public_key_b64": str(
            ready["encryption_public_key_b64"]
        ),
        "device_signing_public_key_b64": str(
            ready["signing_public_key_b64"]
        ),
        "controller_key_id": controller_id,
        "controller_public_key_b64": controller_public_b64,
        "controller_private_pem": pem,
        "saved_at": int(time.time()),
    }
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps(row), encoding="utf-8")
    SESSION_PATH.chmod(0o600)
    return row


def _load_session() -> dict[str, Any] | None:
    try:
        row = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            return None
        private = serialization.load_pem_private_key(
            row["controller_private_pem"].encode(),
            password=None,
        )
        row["_private"] = private
        return row
    except Exception:
        return None


def _clear_session() -> None:
    try:
        SESSION_PATH.unlink()
    except FileNotFoundError:
        pass


def _pair(ready: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    private, controller_id, controller_public = _generate_controller()
    now = int(time.time())
    request = {
        "kind": "phone_pair_request",
        "session_id": str(ready["session_id"]),
        "device_key_id": str(ready["device_key_id"]),
        "controller_key_id": controller_id,
        "controller_public_key_b64": controller_public,
        "ts": now,
        "protocol": 2,
    }
    _publish(str(ready["session_topic"]), request)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ack in reversed(_messages(str(ready["session_topic"]), "2m")):
            if (
                ack.get("kind") != "phone_pair_ack"
                or ack.get("session_id") != ready["session_id"]
                or ack.get("device_key_id") != ready["device_key_id"]
                or ack.get("controller_key_id") != controller_id
            ):
                continue

            signed = _canonical(
                "phone_pair_ack",
                str(ack["session_id"]),
                str(ack["device_key_id"]),
                str(ack["controller_key_id"]),
                str(ack["wrapped_key_b64"]),
                str(ack["nonce_b64"]),
                str(ack["ciphertext_b64"]),
                str(ack["ts"]),
            )
            if not _verify(
                str(ready["signing_public_key_b64"]),
                signed,
                str(ack.get("device_signature_b64") or ""),
            ):
                continue

            plain = json.loads(_hybrid_decrypt(private, ack).decode("utf-8"))
            if plain.get("ok") is not True:
                continue
            row = _save_session(
                ready,
                private,
                controller_id,
                controller_public,
            )
            row["_private"] = private
            return row
        time.sleep(0.7)
    raise TimeoutError("Airi-PC non e riuscita ad abbinarsi automaticamente.")


def _session(timeout: float = 30.0, force_new: bool = False) -> dict[str, Any]:
    if not force_new:
        cached = _load_session()
        if cached is not None:
            return cached
    ready = _discover_ready(timeout=min(timeout, 30.0))
    return _pair(ready, timeout=min(timeout, 30.0))


def _verify_result(session: dict[str, Any], packet: dict[str, Any]) -> bool:
    try:
        signed = _canonical(
            "phone_result",
            str(packet["session_id"]),
            str(packet["device_key_id"]),
            str(packet["controller_key_id"]),
            str(packet["command_id"]),
            str(packet["wrapped_key_b64"]),
            str(packet["nonce_b64"]),
            str(packet["ciphertext_b64"]),
            str(packet["ts"]),
        )
        return _verify(
            str(session["device_signing_public_key_b64"]),
            signed,
            str(packet.get("device_signature_b64") or ""),
        )
    except Exception:
        return False


def _send_once(
    session: dict[str, Any],
    op: str,
    args: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    private = session["_private"]
    command_id = uuid.uuid4().hex[:20]
    now = int(time.time())
    inner = {
        "id": command_id,
        "op": op,
        "args": args or {},
        "issued_at": now,
        "expires_at": now + min(120, max(20, int(timeout) + 10)),
    }
    envelope = _hybrid_encrypt(
        str(session["device_encryption_public_key_b64"]),
        json.dumps(inner, separators=(",", ":")).encode(),
    )
    ts = int(time.time())
    signed = _canonical(
        "phone_cmd",
        str(session["session_id"]),
        str(session["device_key_id"]),
        str(session["controller_key_id"]),
        envelope["wrapped_key_b64"],
        envelope["nonce_b64"],
        envelope["ciphertext_b64"],
        str(ts),
    )
    packet = {
        "kind": "phone_cmd",
        "session_id": session["session_id"],
        "device_key_id": session["device_key_id"],
        "controller_key_id": session["controller_key_id"],
        **envelope,
        "ts": ts,
        "controller_signature_b64": _sign(private, signed),
        "protocol": 2,
    }
    _publish(str(session["session_topic"]), packet)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for result in reversed(_messages(str(session["session_topic"]), "2m")):
            if (
                result.get("kind") != "phone_result"
                or result.get("command_id") != command_id
                or result.get("session_id") != session["session_id"]
                or result.get("device_key_id") != session["device_key_id"]
                or result.get("controller_key_id")
                != session["controller_key_id"]
                or not _verify_result(session, result)
            ):
                continue
            plain = json.loads(
                _hybrid_decrypt(private, result).decode("utf-8")
            )
            return plain
        time.sleep(0.7)
    raise TimeoutError(f"phone command timed out: {op}")


def send(
    op: str,
    args: dict[str, Any] | None = None,
    *,
    timeout: float = 45.0,
) -> dict[str, Any]:
    session = _session(timeout=timeout)
    try:
        result = _send_once(session, op, args, timeout)
    except TimeoutError:
        _clear_session()
        session = _session(timeout=timeout, force_new=True)
        result = _send_once(session, op, args, timeout)

    if op == "stop" and result.get("ok"):
        _clear_session()
    return result


def screenshot(path: str, *, timeout: float = 60.0) -> dict[str, Any]:
    result = send("screenshot", timeout=timeout)
    if not result.get("ok"):
        return result
    url = result.get("frame_url")
    if not url:
        raise RuntimeError("phone returned no frame URL")
    req = urllib.request.Request(
        str(url),
        headers={"User-Agent": "Airi-PC-Phone-Control/0.2"},
    )
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
    parser.add_argument("--timeout", type=float, default=45.0)
    sub = parser.add_subparsers(dest="command", required=True)

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
        if ns.command == "tap":
            result = send(
                "tap",
                {"x": ns.x, "y": ns.y},
                timeout=ns.timeout,
            )
        elif ns.command == "swipe":
            result = send(
                "swipe",
                {
                    "x1": ns.x1,
                    "y1": ns.y1,
                    "x2": ns.x2,
                    "y2": ns.y2,
                    "duration_ms": ns.duration_ms,
                },
                timeout=ns.timeout,
            )
        elif ns.command == "text":
            result = send(
                "text",
                {"text": ns.value},
                timeout=ns.timeout,
            )
        elif ns.command == "screenshot":
            result = screenshot(ns.path, timeout=ns.timeout)
        else:
            result = send(ns.command, timeout=ns.timeout)

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok", True) else 2
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
