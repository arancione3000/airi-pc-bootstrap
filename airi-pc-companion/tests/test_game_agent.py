from game_agent.core import GameAgent,GameProfile,GameState,ActionEngine
from game_agent.safety import SafetyController

def test_profile_and_safety():
 p=GameProfile('offline',frame_fps=5.0); assert p.frame_fps==5.0
 s=SafetyController(); assert not s.can_act(); s.enable(True); assert s.can_act(); s.stop_all(); assert not s.can_act()

def test_game_loop():
 class B:
  def move_mouse(self,x,y): pass
  def click(self,x,y): pass
  def press(self,k): pass
 b=B(); a=ActionEngine(b)
 agent=GameAgent(lambda:{}, lambda f:GameState(1.0,objective='test'), a)
 out=agent.run_once(lambda st:('click',{'x':1,'y':2})); assert out['result']['success']
