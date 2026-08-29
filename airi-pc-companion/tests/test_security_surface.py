from pathlib import Path

def test_source_has_no_obvious_secret_literals():
 root=Path(__file__).resolve().parents[1]
 for p in root.rglob('*'):
  if not p.is_file() or '.git' in p.parts or p.name == 'test_security_surface.py': continue
  if p.suffix in {'.py','.ps1','.sh','.json','.md','.yml','.yaml'}:
   text=p.read_text(encoding='utf-8',errors='ignore')
   assert 'sk-' not in text.lower()
   assert 'OPENROUTER_API_KEY=' not in text

def test_app_does_not_store_pairing_token_in_config():
 text=(Path(__file__).resolve().parents[1]/'app/main.py').read_text()
 assert "json.dumps({k:v for k,v in d.items() if k!='token'}" in text
