from __future__ import annotations
from threading import Event
class SafetyController:
    def __init__(self): self.stop_event=Event(); self.enabled=False
    def enable(self,confirmed=False):
        if not confirmed: raise PermissionError('game automation requires explicit enable')
        self.stop_event.clear(); self.enabled=True
    def stop_all(self): self.stop_event.set(); self.enabled=False
    def can_act(self): return self.enabled and not self.stop_event.is_set()
