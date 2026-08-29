from __future__ import annotations
import json
from pathlib import Path
from .core import GameProfile
class ProfileStore:
    def __init__(self, root: Path): self.root=Path(root)
    def load(self,name):
        d=json.loads((self.root/f'{name}.json').read_text(encoding='utf-8')); return GameProfile(**d)
    def save(self,profile):
        self.root.mkdir(parents=True,exist_ok=True); (self.root/f'{profile.name}.json').write_text(json.dumps(profile.__dict__,indent=2),encoding='utf-8')
