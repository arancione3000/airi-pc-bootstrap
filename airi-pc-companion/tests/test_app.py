from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"

def test_app_files_exist():
 assert APP.exists()
 assert "airi_url" in APP.read_text()
