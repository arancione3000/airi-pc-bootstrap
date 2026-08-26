from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Optional
import time
from coding import (analyze, read, search, write, patch, test, build, lint, shell,
                    git_status, git_diff, git_log, git_commit, project_context,
                    load_project_context, scope_check, snapshot, restore_snapshot,
                    diff_summary, guardrail_check)
from skills import (list_skills, load_skill, create_skill, update_skill, test_skill,
                    delete_skill, memory_read, memory_update, session_event,
                    task_start, task_read, task_update, task_finish)

MAX_ATTEMPTS = 5

def _project_dir(path: str) -> str:
    return str(Path(path).parent)

def apply_fix(path, old, new, test_command='', declared_scope=None, max_attempts=5):
    if not 1 <= int(max_attempts) <= MAX_ATTEMPTS: raise ValueError('max_attempts must be 1..5')
    ctx = load_project_context('.')
    scope_check([path], declared_scope or [path])
    snap = snapshot([path], label=f'apply-fix:{path}')
    history=[]
    for attempt in range(1, int(max_attempts)+1):
        try:
            result = patch(path, old, new, declared_scope=declared_scope or [path])
            vr = test(test_command, _project_dir(path)) if test_command else {'returncode':0,'stdout':'no test command','stderr':''}
            history.append({'attempt':attempt,'verification':vr})
            if vr.get('returncode') == 0:
                return {'ok':True,'rolled_back':False,'attempts':attempt,'context':ctx,'snapshot':snap,'patch':result,'verification':vr,'history':history}
        except Exception as exc:
            history.append({'attempt':attempt,'error':str(exc)})
        restore_snapshot(snap['snapshot'])
    return {'ok':False,'rolled_back':True,'attempts':int(max_attempts),'context':ctx,'snapshot':snap,'history':history,'verification':history[-1].get('verification') if history else None}

def verify_change(path, test_command):
    ctx = load_project_context('.')
    return {'path':path,'context':ctx,'verification':test(test_command,_project_dir(path))}

def plan(goal, project_path='.', steps: Optional[Iterable[str]]=None, scope: Optional[Iterable[str]]=None):
    ctx = load_project_context(project_path)
    todo = task_start(goal, steps or ['analyze context','inspect code','edit safely','test and verify','review diff','persist'], scope=scope)
    return {'goal':goal,'project':analyze(project_path),'context':ctx,'skills':list_skills(),
            'memory':memory_read(),'todo':todo,
            'policy':'context required; declared scope required for autonomous changes; max 5 repair attempts; failed verification restores snapshot; diff+guardrails precede commits; tests/security cannot be weakened.'}

def autonomous_change_cycle(changes: list[dict[str, Any]], project_path='.', test_command='', declared_scope=None, max_attempts=5):
    ctx=load_project_context(project_path)
    if not 1 <= int(max_attempts) <= MAX_ATTEMPTS: raise ValueError('max_attempts must be 1..5')
    if not changes: raise ValueError('changes must not be empty')
    scope=declared_scope or [c['path'] for c in changes]
    scope_check([c['path'] for c in changes], scope)
    steps=[f"change: {c['path']}" for c in changes] + ['full verification','diff review','persist']
    todo=task_start('autonomous coding change cycle',steps,scope)
    records=[]
    for idx,c in enumerate(changes):
        path=c['path']; snap=snapshot([path],label=f'agent:{path}')
        succeeded=False; attempts=[]
        old_content=read(path)['content'] if Path(path).exists() else None
        for attempt in range(1,int(max_attempts)+1):
            try:
                if 'old' in c and 'new' in c:
                    patch(path,c['old'],c['new'],declared_scope=scope)
                else:
                    write(path,c['content'],declared_scope=scope)
                vr=test(c.get('test_command') or test_command, project_path) if (c.get('test_command') or test_command) else {'returncode':0,'stdout':'no test command','stderr':''}
                attempts.append({'attempt':attempt,'returncode':vr.get('returncode')})
                if vr.get('returncode') == 0:
                    succeeded=True; task_update(idx,'done',f'passed on attempt {attempt}'); break
            except Exception as exc:
                attempts.append({'attempt':attempt,'error':str(exc)})
            restore_snapshot(snap['snapshot'])
        if not succeeded:
            restore_snapshot(snap['snapshot']); task_update(idx,'failed','max attempts reached; snapshot restored')
            task_finish('failed',f'rollback for {path}')
            return {'ok':False,'rolled_back':True,'context':ctx,'todo':task_read(),'changes':records+[{'path':path,'attempts':attempts}]}
        records.append({'path':path,'snapshot':snap['snapshot'],'attempts':attempts,'before':old_content})
    verification=test(test_command,project_path) if test_command else {'returncode':0,'stdout':'no full test command','stderr':''}
    if verification.get('returncode') != 0:
        for rec in reversed(records): restore_snapshot(rec['snapshot'])
        task_finish('failed','full verification failed; all task snapshots restored')
        return {'ok':False,'rolled_back':True,'context':ctx,'todo':task_read(),'changes':records,'verification':verification}
    task_update(len(changes),'done','full verification passed')
    review=diff_summary(project_path,scope)
    guard=guardrail_check(project_path,scope)
    if not guard['ok']:
        for rec in reversed(records): restore_snapshot(rec['snapshot'])
        task_finish('failed','guardrail violation; all task snapshots restored')
        return {'ok':False,'rolled_back':True,'context':ctx,'todo':task_read(),'changes':records,'verification':verification,'diff':review,'guardrails':guard}
    task_update(len(changes)+1,'done','diff and guardrails passed')
    finished=task_finish('done','autonomous coding cycle complete; ready for explicit persistence step')
    return {'ok':True,'rolled_back':False,'context':ctx,'todo':finished,'changes':records,'verification':verification,'diff':review,'guardrails':guard}

def prepare_commit(project_path='.', declared_scope=None):
    ctx=load_project_context(project_path)
    review=diff_summary(project_path,declared_scope)
    if not review.get('available'): raise RuntimeError('Git repository unavailable')
    if not review['guardrails']['ok']: raise PermissionError('; '.join(review['guardrails']['violations']))
    return {'ready':True,'context':ctx,'review':review}

def atomic_commit(message, project_path='.', declared_scope=None, allow_test_changes=False, allow_security_changes=False):
    prepared=prepare_commit(project_path,declared_scope)
    result=git_commit(message,project_path,declared_scope,allow_test_changes,allow_security_changes)
    session_event('git-commit','committed' if result.get('committed') else 'noop',prepared['review']['summary'],result.get('commit_sha'))
    return {'ok':result.get('ok',False),'prepared':prepared,'commit':result}

def agent(goal, project_path='.', max_attempts=5, steps=None, scope=None, changes=None, test_command=''):
    attempts=min(MAX_ATTEMPTS,max(1,int(max_attempts)))
    if not changes:
        p=plan(goal,project_path,steps,scope)
        return {'ok':True,'mode':'coding-agent-orchestrator','max_attempts':attempts,'plan':p,
                'capabilities':['context','todo','structured-search','scoped-edit','snapshot-rollback','diff-summary','guardrails','atomic-commit'],
                'note':'No concrete changes supplied; plan created and context loaded, no source was modified.'}
    ctx=load_project_context(project_path)
    cycle=autonomous_change_cycle(changes,project_path,test_command,scope,attempts)
    return {'ok':cycle.get('ok',False),'mode':'coding-agent-orchestrator','max_attempts':attempts,
            'plan':{'goal':goal,'project':analyze(project_path),'context':ctx,'scope':list(scope or []),'steps':steps or [c['path'] for c in changes]},
            'cycle':cycle}
