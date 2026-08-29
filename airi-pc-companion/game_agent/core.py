from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable
import time, uuid

@dataclass
class GameProfile:
    name: str
    executable: list[str] = field(default_factory=list)
    input_mapping: dict[str, Any] = field(default_factory=dict)
    screen_region: dict[str, int] | None = None
    frame_fps: float = 5.0
    safety: dict[str, Any] = field(default_factory=lambda: {'allow_automation': False})

@dataclass
class GameState:
    timestamp: float
    player_position: Any = None
    camera_orientation: Any = None
    visible_objects: list[Any] = field(default_factory=list)
    enemies: list[Any] = field(default_factory=list)
    objective: Any = None
    ui_state: Any = None
    health: Any = None
    ammo: Any = None
    resources: dict[str, Any] = field(default_factory=dict)
    menu_state: Any = None
    loading: Any = None
    game_over: Any = None
    confidence: float = 0.0

@dataclass
class ActionResult:
    action_id: str
    success: bool
    reason: str = ''
    started_at: float = field(default_factory=time.time)
    finished_at: float = field(default_factory=time.time)

class ActionEngine:
    def __init__(self, backend: Any, stop_event=None): self.backend=backend; self.stop_event=stop_event
    def _call(self, name: str, *a, **kw):
        aid=str(uuid.uuid4()); start=time.time()
        if self.stop_event and self.stop_event.is_set(): return ActionResult(aid,False,'cancelled',start,time.time())
        try: getattr(self.backend,name)(*a,**kw); return ActionResult(aid,True,'',start,time.time())
        except Exception as e: return ActionResult(aid,False,type(e).__name__,start,time.time())
    def move_mouse(self,x,y): return self._call('move_mouse',x,y)
    def click(self,x,y): return self._call('click',x,y)
    def press(self,key): return self._call('press',key)
    def hold(self,key): return self._call('key_down',key)
    def release(self,key): return self._call('key_up',key)
    def wait(self,seconds): return self._call('wait',seconds)

class GameAgent:
    def __init__(self, capture: Callable[[], Any], observe: Callable[[Any], GameState], actions: ActionEngine):
        self.capture=capture; self.observe=observe; self.actions=actions
    def step(self) -> GameState:
        frame=self.capture(); return self.observe(frame)
    def run_once(self, decision: Callable[[GameState], tuple[str,dict[str,Any]]]) -> dict[str,Any]:
        state=self.step(); action,kwargs=decision(state)
        result=self.actions._call(action,**kwargs)
        return {'state':asdict(state),'action':action,'result':asdict(result)}
