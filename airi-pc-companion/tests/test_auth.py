from pathlib import Path
from companion.auth import AuthStore

def test_auth_roundtrip(tmp_path: Path):
 a=AuthStore(tmp_path); assert a.initial_token; assert a.verify(a.initial_token); assert not a.verify('bad'); b=a.rotate(); assert a.verify(b); assert not a.verify(a.initial_token)
