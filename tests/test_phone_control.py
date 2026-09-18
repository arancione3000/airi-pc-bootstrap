from __future__ import annotations

import base64
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "computer"))
from control_plane import phone_control as pc


def test_hybrid_envelope_roundtrip():
    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_der = private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_b64 = base64.b64encode(public_der).decode()
    plain = b'{"hello":"phone"}'
    packet = pc._hybrid_encrypt(public_b64, plain)
    assert pc._hybrid_decrypt(private, packet) == plain


def test_controller_signature_roundtrip():
    private, key_id, public_b64 = pc._generate_controller()
    data = pc._canonical("phone_cmd", "session", "device")
    signature = pc._sign(private, data)
    assert pc._verify(public_b64, data, signature)
    assert not pc._verify(public_b64, data + b"x", signature)

    public_der = private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert key_id == pc._key_id(public_der)


def test_ready_self_signature_verification():
    private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    signing_der = private.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signing_b64 = base64.b64encode(signing_der).decode()
    row = {
        "session_id": "session-1",
        "device_key_id": "device-1",
        "encryption_public_key_b64": "enc",
        "signing_public_key_b64": signing_b64,
        "session_topic": "topic-1",
        "ts": 10,
        "expires_at": 30,
    }
    canonical = pc._canonical(
        "phone_ready",
        row["session_id"],
        row["device_key_id"],
        row["encryption_public_key_b64"],
        row["signing_public_key_b64"],
        row["session_topic"],
        str(row["ts"]),
        str(row["expires_at"]),
    )
    row["device_signature_b64"] = base64.b64encode(
        private.sign(canonical, padding.PKCS1v15(), hashes.SHA256())
    ).decode()
    assert pc._verify_ready(row)

    row["session_topic"] = "tampered"
    assert not pc._verify_ready(row)
