import argparse, os
from pathlib import Path
from .auth import AuthStore
from .server import serve
root=Path(os.getenv("AIRIPC_COMPANION_STATE",str(Path.home()/".airi-pc-companion")))
p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd")
run=sub.add_parser("run"); run.add_argument("--host",default=os.getenv("AIRIPC_COMPANION_HOST","127.0.0.1")); run.add_argument("--port",type=int,default=int(os.getenv("AIRIPC_COMPANION_PORT","17894")))
pair=sub.add_parser("pair")
dis=sub.add_parser("disable"); en=sub.add_parser("enable"); rev=sub.add_parser("revoke")
a=p.parse_args(); root.mkdir(parents=True,exist_ok=True)
if a.cmd=="pair":
 s=AuthStore(root)
 if not s.initial_token: print("Already paired. Use 'revoke' then 'pair' to rotate.")
 else: print(s.initial_token)
elif a.cmd=="disable": (root/"DISABLED").touch(mode=0o600,exist_ok=True); print("disabled")
elif a.cmd=="enable": (root/"DISABLED").unlink(missing_ok=True); print("enabled")
elif a.cmd=="revoke": AuthStore(root).revoke(); print("revoked")
else: serve(a.host,a.port,root)
