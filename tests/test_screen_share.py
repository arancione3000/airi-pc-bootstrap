from __future__ import annotations

import base64
import json
import shutil
import subprocess

import pytest

from control_plane.screen_share import _encrypt_descriptor


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl unavailable")
def test_screen_offer_rsa_oaep_roundtrip(tmp_path):
    private = tmp_path / "private.pem"
    public_der = tmp_path / "public.der"
    cipher = tmp_path / "cipher.bin"
    plain = tmp_path / "plain.bin"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:3072", "-out", str(private)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-outform", "DER", "-out", str(public_der)],
        check=True, capture_output=True,
    )
    descriptor = {"u": "https://example.trycloudflare.com/view/test-token", "s": "session-test", "v": 1}
    encrypted = _encrypt_descriptor(base64.b64encode(public_der.read_bytes()).decode("ascii"), descriptor)
    cipher.write_bytes(base64.b64decode(encrypted))
    subprocess.run(
        [
            "openssl", "pkeyutl", "-decrypt", "-inkey", str(private),
            "-in", str(cipher), "-out", str(plain),
            "-pkeyopt", "rsa_padding_mode:oaep",
            "-pkeyopt", "rsa_oaep_md:sha256",
            "-pkeyopt", "rsa_mgf1_md:sha1",
        ],
        check=True, capture_output=True,
    )
    assert json.loads(plain.read_text()) == descriptor
