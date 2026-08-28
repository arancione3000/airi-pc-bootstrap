from __future__ import annotations
from .store import load_json, save_json

FILE='model-routing.json'
DEFAULTS={'simple':'fast','coding':'strong','research':'strong','vision':'vision','review':'independent'}
class ModelRouter:
    def __init__(self): self.state=load_json(FILE, {'version':1,'routes':DEFAULTS.copy(),'providers':{}})
    def register_provider(self,name,capabilities,available=False,cost_class='unknown'):
        self.state['providers'][name]={'name':name,'capabilities':list(capabilities),'available':bool(available),'cost_class':cost_class}; save_json(FILE,self.state); return self.state['providers'][name]
    def choose(self, task_type='simple', complexity='medium', needs_vision=False, prefer_speed=False):
        kind='vision' if needs_vision else ('simple' if prefer_speed or complexity=='low' else task_type)
        target=self.state['routes'].get(kind,self.state['routes']['simple'])
        candidates=[p for p in self.state['providers'].values() if p.get('available') and target in p.get('capabilities',[])]
        return {'route':target,'selected':candidates[0]['name'] if candidates else None,'candidates':[p['name'] for p in candidates]}
    def status(self): return self.state
