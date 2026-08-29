from pathlib import Path

def test_app_files_exist():
 assert Path('app/main.py').exists()
 assert 'airi_url' in Path('app/main.py').read_text()
