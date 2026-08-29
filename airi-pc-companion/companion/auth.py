from __future__ import annotations
import hashlib, hmac, json, os, secrets
from pathlib import Path

class AuthStore:
    def __init__(self, root: Path):
        self.root=root; self.path=root/'auth.json'; root.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            token=secrets.token_urlsafe(32)
            self.path.write_text(json.dumps({'token_hash': hashlib.sha256(token.encode()).hexdigest()}))
            try: os.chmod(self.path,0o600)
            except OSError: pass
            self.initial_token=token
        else: self.initial_token=None
        data=json.loads(self.path.read_text()); self._hash=data['token_hash']

    def verify(self, token: str) -> bool:
        return hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), self._hash)

    def revoke(self) -> None:
        self._hash = ''
        self.path.write_text(json.dumps({'token_hash': ''}))
        try: os.chmod(self.path,0o600)
        except OSError: pass

    def rotate(self) -> str:
        token=secrets.token_urlsafe(32); self._hash=hashlib.sha256(token.encode()).hexdigest()
        self.path.write_text(json.dumps({'token_hash':self._hash}))
        try: os.chmod(self.path,0o600)
        except OSError: pass
        return token
