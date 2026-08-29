from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
@dataclass
class Frame:
    timestamp: float
    image: Any
    region: tuple[int,int,int,int] | None = None
class CaptureBackend:
    def __init__(self, fps=5.0, region=None): self.fps=max(0.5,min(float(fps),30.0)); self.region=region; self._last=0.0
    def capture(self):
        delay=1.0/self.fps; now=time.time()
        if now-self._last<delay: time.sleep(delay-(now-self._last))
        self._last=time.time()
        from PIL import ImageGrab
        img=ImageGrab.grab(bbox=self.region)
        return Frame(time.time(),img,self.region)
