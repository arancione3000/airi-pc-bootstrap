from __future__ import annotations
from typing import Any
from coding import test, build, lint, git_status
from .store import now

class VerificationEngine:
    def run(self, *, requirements: list[str] | None = None, tests: str | None = None,
            build_cmd: str | None = None, lint_cmd: str | None = None,
            project_path: str = '.', runtime: dict[str, Any] | None = None,
            security: dict[str, Any] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {'requirements':'PASS', 'tests':'SKIPPED', 'runtime':'SKIPPED', 'security':'SKIPPED', 'git':'SKIPPED', 'ready':False, 'timestamp':now()}
        if requirements:
            results['requirements']='PASS' if all(bool(x and str(x).strip()) for x in requirements) else 'FAIL'
        if tests:
            r=test(tests,project_path,170); results['tests']='PASS' if r.get('returncode')==0 else 'FAIL'; results['test_result']=r
        if build_cmd:
            r=build(build_cmd,project_path,170); results['build']='PASS' if r.get('returncode')==0 else 'FAIL'; results['build_result']=r
        if lint_cmd:
            r=lint(lint_cmd,project_path,170); results['lint']='PASS' if r.get('returncode')==0 else 'FAIL'; results['lint_result']=r
        if runtime is not None:
            results['runtime']='PASS' if runtime.get('ok') is True else 'PARTIAL' if runtime.get('ok') is not False else 'FAIL'
        if security is not None:
            results['security']='PASS' if security.get('ok') is True else 'PARTIAL' if security.get('ok') is not False else 'FAIL'
        gs=git_status(project_path)
        results['git']='PASS' if gs.get('available') and not gs.get('stdout','').splitlines()[1:] else 'PARTIAL' if gs.get('available') else 'SKIPPED'
        blocking={'FAIL'}
        results['ready']=not any(results.get(k) in blocking for k in ('requirements','tests','build','lint','runtime','security','git'))
        return results
