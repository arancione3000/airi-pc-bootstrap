from __future__ import annotations
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=[
    ROOT/'scripts/airi-next-session', ROOT/'scripts/airi-session-rebuild',
    ROOT/'scripts/airi-rebuild', ROOT/'scripts/airi-connect',
    ROOT/'scripts/airi-chat-bootstrap', ROOT/'scripts/public-bootstrap.sh'
]

def test_bootstrap_scripts_forbid_pipe_to_shell_and_hard_reset():
    forbidden_pipe=re.compile(r'(?:curl|wget)[^\n]*\|\s*(?:sh|bash)')
    for script in SCRIPTS:
        text=script.read_text(encoding='utf-8')
        assert not forbidden_pipe.search(text), script
        assert 'git reset --hard' not in text, script
        assert 'git ls-remote' in text, script
        assert 'EXPECTED_SHA' in text, script

def test_bootstrap_scripts_have_valid_shell_syntax():
    for script in SCRIPTS:
        result=subprocess.run(['sh','-n',str(script)],cwd=ROOT,text=True,capture_output=True,timeout=10)
        assert result.returncode==0, f'{script}: {result.stderr}'
