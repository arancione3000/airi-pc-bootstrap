from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "computer"))
from control_plane import phone_control as pc


def test_pair_secret_roundtrip(tmp_path, monkeypatch):
    secret = os.urandom(32)
    text = base64.urlsafe_b64encode(secret).decode().rstrip("=")
    target = tmp_path / "pair"
    monkeypatch.setattr(pc, "PAIR_PATH", target)
    out = pc.save_pair_secret(text)
    assert out["ok"] is True
    assert pc.pair_secret() == secret
    assert target.read_text().strip() == text


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


def test_auth_tag_is_stable_and_secret_bound():
    a = b"a" * 32
    b = b"b" * 32
    first = pc._tag(a, "phone_cmd", "1", "2")
    assert first == pc._tag(a, "phone_cmd", "1", "2")
    assert first != pc._tag(b, "phone_cmd", "1", "2")
