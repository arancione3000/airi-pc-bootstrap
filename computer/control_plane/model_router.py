from __future__ import annotations
from .store import load_json, save_json
FILE="model-routing.json"
DEFAULTS={"simple":"chatgpt","coding":"chatgpt","research":"chatgpt","vision":"chatgpt","review":"chatgpt"}
CHATGPT_PROVIDER={"name":"chatgpt","capabilities":["simple","coding","research","vision","review"],"available":True,"cost_class":"reasoning-authority"}
class ModelRouter:
    """Represent the reasoning boundary without selecting a second LLM."""
    def __init__(self):
        self.state=load_json(FILE,{})
        self.state["version"]=2; self.state["routing_authority"]="chatgpt"; self.state["routes"]=DEFAULTS.copy(); self.state["providers"]={"chatgpt":dict(CHATGPT_PROVIDER)}; save_json(FILE,self.state)
    def register_provider(self,name,capabilities=None,available=False,cost_class="unknown"):
        if name!="chatgpt": return {"name":str(name),"capabilities":list(capabilities or []),"available":False,"cost_class":"disabled","disabled":True,"reason":"ChatGPT is the sole reasoning authority."}
        self.state["providers"]={"chatgpt":dict(CHATGPT_PROVIDER)}; save_json(FILE,self.state); return self.state["providers"]["chatgpt"]
    def choose(self,task_type="simple",complexity="medium",needs_vision=False,prefer_speed=False):
        del complexity,prefer_speed
        kind="vision" if needs_vision else task_type; route=self.state["routes"].get(kind,"chatgpt")
        return {"route":route,"selected":"chatgpt","candidates":["chatgpt"],"reasoning_authority":"chatgpt"}
    def status(self): return self.state
