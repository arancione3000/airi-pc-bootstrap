from __future__ import annotations
LEVELS={'READ_ONLY':0,'LOW_RISK':1,'HIGH_RISK':2,'DESTRUCTIVE':3}
CAPABILITY_LEVEL={
 'screen':0,'system':0,'windows':0,'processes':0,
 'mouse':1,'keyboard':1,'applications':1,
 'filesystem':2,'process_control':2,
 'shutdown':3,'reboot':3,'delete':3,
}

def authorize(capability: str, action: str, requested_level: str|None=None, confirmed: bool=False) -> tuple[bool,str]:
    level=requested_level or next((k for k,v in LEVELS.items() if v==CAPABILITY_LEVEL.get(capability,3)),'DESTRUCTIVE')
    if level not in LEVELS: return False,'unknown_permission_level'
    if capability in {'shutdown','reboot','delete'} and not confirmed: return False,'human_confirmation_required'
    if LEVELS[level]>=LEVELS['HIGH_RISK'] and not confirmed: return False,'human_confirmation_required'
    return True,level
