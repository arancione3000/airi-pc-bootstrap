from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _inside(path: object, root: Path) -> bool:
    if isinstance(path, int):
        return True
    try:
        candidate = Path(os.fsdecode(path)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
        candidate = candidate.resolve(strict=False)
    except Exception:
        return False
    return candidate == root or root in candidate.parents


def sandbox_violation(event: str, args, root: Path) -> str | None:
    root = Path(root).resolve()
    blocked_network = {
        "socket.connect", "socket.connect_ex", "socket.bind", "socket.listen",
        "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
        "socket.gethostbyname", "socket.gethostbyaddr",
    }
    if event in blocked_network or event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
        return f"Evolution Lab sandbox blocked capability: {event}"

    if event == "open" and args:
        path = args[0]
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        wants_write = False
        if isinstance(mode, str):
            wants_write = any(ch in mode for ch in ("w", "a", "x", "+"))
        if isinstance(flags, int):
            wants_write = wants_write or bool(
                flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                )
            )
        if wants_write and not _inside(path, root):
            return f"Evolution Lab blocked write outside sandbox: {path}"

    if event in {"os.remove", "os.rmdir", "os.mkdir", "os.chdir"} and args:
        if not _inside(args[0], root):
            return f"Evolution Lab blocked filesystem mutation outside sandbox: {args[0]}"

    if event in {"os.rename", "os.replace"} and len(args) >= 2:
        if not _inside(args[0], root) or not _inside(args[1], root):
            return "Evolution Lab blocked rename outside sandbox"
    return None


def install_sandbox_guard(root: Path) -> None:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def guard(event: str, args):
        violation = sandbox_violation(event, args, root)
        if violation:
            raise PermissionError(violation)

    sys.addaudithook(guard)


def main() -> int:
    from . import lab

    root = lab.LAB_STATE.resolve()
    tmp = root / "tmp"
    cache = root / "cache"
    tmp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    install_sandbox_guard(root)

    try:
        import torch
        torch.set_num_threads(max(1, min(2, int(os.environ.get("AIRI_LAB_CPU_THREADS", "2")))))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass

    result = lab.run_cycle(
        population=int(os.environ.get("AIRI_LAB_POPULATION", "6")),
        generations=int(os.environ.get("AIRI_LAB_GENERATIONS", "2")),
        candidate_epochs=int(os.environ.get("AIRI_LAB_CANDIDATE_EPOCHS", "1")),
        finalist_epochs=int(os.environ.get("AIRI_LAB_FINALIST_EPOCHS", "2")),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
