from __future__ import annotations
import re, shutil, time
from pathlib import Path
from typing import Optional
from coding import ROOT, SKILLS, AI, safe_path, write, read

def _name(name):
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,80}',name):raise ValueError('invalid skill name')
    return name

def list_skills():
    SKILLS.mkdir(parents=True,exist_ok=True);out=[]
    for d in sorted(SKILLS.iterdir()):
        if d.is_dir() and (d/'SKILL.md').exists():out.append({'name':d.name,'path':str((d/'SKILL.md').relative_to(ROOT)),'content':(d/'SKILL.md').read_text(errors='replace')[:50000]})
    return {'count':len(out),'skills':out}

def load_skill(name):
    _name(name);return read(str(Path('skills')/name/'SKILL.md'))

def create_skill(name,description,instructions,tools=None):
    _name(name);d=SKILLS/name;d.mkdir(parents=True,exist_ok=True)
    text=f'# {name}\n\n## Description\n{description}\n\n## Instructions\n{instructions}\n\n## Tools\n'+''.join(f'- {x}\n' for x in (tools or []))
    (d/'SKILL.md').write_text(text);return {'ok':True,'name':name,'path':str((d/'SKILL.md').relative_to(ROOT))}

def update_skill(name,content):
    _name(name);return write(str(Path('skills')/name/'SKILL.md'),content)

def test_skill(name):
    _name(name);p=SKILLS/name/'SKILL.md'
    if not p.exists():raise FileNotFoundError(name)
    text=p.read_text(errors='replace');return {'ok':True,'name':name,'bytes':len(text.encode()),'sections':[x for x in ['Description','Instructions','Tools'] if f'## {x}' in text]}

def delete_skill(name):
    _name(name);d=safe_path(Path('skills')/name);trash=AI/'skill-trash'/f'{name}-{int(time.time())}';trash.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(d,trash);shutil.rmtree(d);return {'ok':True,'name':name,'backup':str(trash.relative_to(ROOT))}

def memory_read():
    AI.mkdir(parents=True,exist_ok=True);p=AI/'PROJECT_MEMORY.md'
    if not p.exists():p.write_text('# Airi-PC Project Memory\n')
    return read(str(p.relative_to(ROOT)))

def memory_update(entry):
    p=AI/'PROJECT_MEMORY.md';m=memory_read()['content'];p.write_text(m.rstrip()+f'\n\n{entry.strip()}\n');return memory_read()
