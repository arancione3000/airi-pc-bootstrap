#!/usr/bin/env python3
from pathlib import Path
import os, runpy

root = Path(os.environ.get("AIRIPC_WORKSPACE_ROOT", Path(__file__).resolve().parents[1]))
os.environ.setdefault("AIRIPC_WORKSPACE_ROOT", str(root))
os.environ.setdefault("DISPLAY", ":99")
runpy.run_path(str(root / "computer" / "airi_agent.py"), run_name="__main__")
