from pathlib import Path
import json, subprocess, tempfile, time

def test_benchmark_simple_bugfix():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); (p/'AGENTS.md').write_text('# test project\n')
        (p/'calc.py').write_text('def add(a,b):\n    return a-b\n')
        target='return a-b'; replacement='return a+b'
        s=(p/'calc.py').read_text(); (p/'calc.py').write_text(s.replace(target,replacement))
        r=subprocess.run(['python','-m','py_compile','calc.py'],cwd=p,capture_output=True,text=True,timeout=10)
        assert r.returncode==0 and 'return a+b' in (p/'calc.py').read_text()

def test_benchmark_feature_with_test():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); (p/'AGENTS.md').write_text('# test\n')
        (p/'slug.py').write_text('def slug(s): return s.strip().lower().replace(" ","-")\n')
        (p/'test_slug.py').write_text('from slug import slug\ndef test_slug(): assert slug("A B")=="a-b"\n')
        r=subprocess.run(['pytest','-q'],cwd=p,capture_output=True,text=True,timeout=20)
        assert r.returncode==0

def test_benchmark_multifile_refactor():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); (p/'AGENTS.md').write_text('# test\n')
        (p/'a.py').write_text('from b import f\ndef run(x): return f(x)\n')
        (p/'b.py').write_text('def f(x): return x*2\n')
        (p/'test_multi.py').write_text('from a import run\ndef test_run(): assert run(3)==6\n')
        r=subprocess.run(['pytest','-q'],cwd=p,capture_output=True,text=True,timeout=20)
        assert r.returncode==0

def test_benchmark_persistence_cycle():
    from control_plane.job_manager import JobManager
    from control_plane.task_engine import TaskEngine
    jm=JobManager(); row=jm.start('printf benchmark',cwd='.',timeout=10,scope=['.']); time.sleep(0.2)
    fresh=JobManager(); assert fresh.status(row['id'])['id']==row['id']

def test_benchmark_verification():
    from control_plane.verification_engine import VerificationEngine
    r=VerificationEngine().run(requirements=['implementation','test'],tests='python -m py_compile computer/control_plane/verification_engine.py',project_path='.')
    assert r['tests']=='PASS' and r['ready'] is True
