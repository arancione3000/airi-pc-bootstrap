from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'computer'))
from security import safe_path, validate_delete_target, shell_command


def test_path_traversal_is_rejected():
    for value in ('../etc/passwd','/etc/passwd'):
        try: safe_path(value)
        except PermissionError: pass
        else: raise AssertionError(value)


def test_workspace_root_delete_is_rejected():
    try: validate_delete_target('.')
    except PermissionError: pass
    else: raise AssertionError('workspace root delete accepted')


def test_dangerous_shell_is_rejected_without_explicit_override():
    for cmd in ('rm -rf /','shutdown -h now','dd if=/dev/zero'):
        try: shell_command(cmd, allow_shell=False)
        except PermissionError: pass
        else: raise AssertionError(cmd)


def test_benign_shell_is_allowed():
    assert shell_command('printf AIRI_SECURITY_OK', allow_shell=False)=='printf AIRI_SECURITY_OK'
