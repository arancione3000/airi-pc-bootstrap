from __future__ import annotations
import json, re, shutil, time
from pathlib import Path
from typing import Iterable, Optional
from coding import ROOT, SKILLS, AI, safe_path, write, read

TASK_STATE = AI / 'task_state.json'
TASK_STATUSES = {'todo','in_progress','done','blocked','failed'}

def _name(name):
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}', name): raise ValueError('invalid skill name')
    return name

def list_skills():
    SKILLS.mkdir(parents=True, exist_ok=True); out=[]
    for d in sorted(SKILLS.iterdir()):
        if d.is_dir() and (d/'SKILL.md').exists():
            out.append({'name': d.name, 'path': str((d/'SKILL.md').relative_to(ROOT)), 'content': (d/'SKILL.md').read_text(errors='replace')[:50000]})
    return {'count': len(out), 'skills': out}

def load_skill(name):
    _name(name); return read(str(Path('skills')/name/'SKILL.md'))

def create_skill(name, description, instructions, tools=None):
    _name(name); d=SKILLS/name; d.mkdir(parents=True, exist_ok=True)
    text=f'# {name}\n\n## Description\n{description}\n\n## Instructions\n{instructions}\n\n## Tools\n'+''.join(f'- {x}\n' for x in (tools or []))
    (d/'SKILL.md').write_text(text); return {'ok':True,'name':name,'path':str((d/'SKILL.md').relative_to(ROOT))}

def update_skill(name, content):
    _name(name); return write(str(Path('skills')/name/'SKILL.md'), content)

def test_skill(name):
    _name(name); p=SKILLS/name/'SKILL.md'
    if not p.exists(): raise FileNotFoundError(name)
    text=p.read_text(errors='replace')
    return {'ok':True,'name':name,'bytes':len(text.encode()),'sections':[x for x in ['Description','Instructions','Tools'] if f'## {x}' in text]}

def delete_skill(name):
    _name(name); d=safe_path(Path('skills')/name); trash=AI/'skill-trash'/f'{name}-{int(time.time())}'
    if not d.exists(): raise FileNotFoundError(name)
    trash.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(d,trash); shutil.rmtree(d)
    return {'ok':True,'name':name,'backup':str(trash.relative_to(ROOT))}

def memory_read():
    AI.mkdir(parents=True, exist_ok=True); p=AI/'PROJECT_MEMORY.md'
    if not p.exists(): p.write_text('# Airi-PC Project Memory\n')
    return read(str(p.relative_to(ROOT)))

def memory_update(entry):
    p=AI/'PROJECT_MEMORY.md'; m=memory_read()['content']
    p.write_text(m.rstrip()+f'\n\n{entry.strip()}\n')
    return memory_read()

def session_event(event: str, outcome='info', details='', commit_sha=None):
    stamp=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    line=f'### {stamp} — {outcome.upper()} — {event}\n{details.strip()}\n'
    if commit_sha: line += f'Commit: `{commit_sha}`\n'
    return memory_update(line)

def task_start(goal: str, steps: Iterable[str], scope: Optional[Iterable[str]]=None):
    steps=list(steps)
    if not steps: raise ValueError('at least one task step required')
    for s in steps:
        if not str(s).strip(): raise ValueError('empty task step')
    AI.mkdir(parents=True, exist_ok=True)
    state={'goal':goal,'scope':list(scope or []),'created_at':time.time(),'updated_at':time.time(),
           'current':0,'steps':[{'title':str(s),'status':'todo','note':''} for s in steps]}
    state['steps'][0]['status']='in_progress'
    TASK_STATE.write_text(json.dumps(state, indent=2))
    session_event('task-start', 'started', f"Goal: {goal}\nSteps: {len(steps)}")
    return state

def task_read():
    if not TASK_STATE.exists(): return {'active':False,'steps':[]}
    state=json.loads(TASK_STATE.read_text())
    state['active']=state.get('final_outcome') is None
    return state

def task_update(index: int, status: str, note=''):
    if status not in TASK_STATUSES: raise ValueError(f'invalid task status: {status}')
    state=task_read()
    if not state.get('active'): raise RuntimeError('no active task')
    if index < 0 or index >= len(state['steps']): raise IndexError(index)
    state['steps'][index].update({'status':status,'note':note})
    if status == 'done' and index + 1 < len(state['steps']):
        for i in range(index+1,len(state['steps'])):
            if state['steps'][i]['status'] == 'todo':
                state['steps'][i]['status']='in_progress'; state['current']=i; break
    state['updated_at']=time.time(); TASK_STATE.write_text(json.dumps(state, indent=2))
    session_event('task-step', 'progress', f"Step {index+1}/{len(state['steps'])}: {status}. {note}")
    return state

def task_finish(outcome='done', note=''):
    if outcome not in {'done','failed','blocked'}: raise ValueError(outcome)
    state=task_read()
    if not state.get('active'): raise RuntimeError('no active task')
    if outcome == 'done':
        for step in state['steps']: step['status']='done'
    state['final_outcome']=outcome; state['final_note']=note; state['updated_at']=time.time()
    TASK_STATE.write_text(json.dumps(state, indent=2))
    session_event('task-finish', outcome, note)
    return state
