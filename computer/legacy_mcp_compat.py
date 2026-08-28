from __future__ import annotations
import inspect

def patch_legacy_mcp(module):
    original = module.mcp
    try:
        src = inspect.getsource(original)
    except (OSError, TypeError):
        return original
    marker = "if path=='control_plane':"
    end = "elif path=='status': result=status()"
    start = src.find(marker)
    stop = src.find(end, start) if start >= 0 else -1
    if start < 0 or stop < 0:
        return original
    block = src[start:stop]
    block = block.replace("action=payload.get('action','status')", "cp_action=payload.get('action','status')")
    block = block.replace("if action==", "if cp_action==").replace("elif action==", "elif cp_action==")
    fixed_src = src[:start] + block + src[stop:]
    ns = dict(module.__dict__)
    code = compile(fixed_src, inspect.getsourcefile(original) or '<airi-compat>', 'exec')
    exec(code, ns)
    fixed = ns.get(original.__name__)
    if fixed is not None:
        module.mcp = fixed
        return fixed
    return original
