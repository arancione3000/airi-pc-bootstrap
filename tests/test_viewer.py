from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("viewer", ROOT / "computer" / "viewer.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_viewer_has_private_token_and_safe_actions():
    assert viewer.build_router is not None
    src = (ROOT / "computer" / "viewer.py").read_text(encoding="utf-8")
    assert "X-Airi-Viewer-Token" in src
    assert "viewer action not allowed" in src


def test_viewer_never_accepts_shell_action():
    src = (ROOT / "computer" / "viewer.py").read_text(encoding="utf-8")
    assert "run_shell" not in src.split("allowed =",1)[1].split("}",1)[0]
