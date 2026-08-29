from __future__ import annotations
from .core import GameState
class StateEstimator:
    def __init__(self, detector=None): self.detector=detector
    def estimate(self, frame):
        if self.detector is None: return GameState(timestamp=frame.timestamp)
        data=self.detector(frame.image) or {}; return GameState(timestamp=frame.timestamp, **{k:v for k,v in data.items() if k in GameState.__dataclass_fields__})
