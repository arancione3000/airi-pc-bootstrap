from __future__ import annotations
import os, shutil, subprocess, time, urllib.request
from .store import load_json, save_json, now
from .reliability import REGISTRY

BASE = __import__('pathlib').Path(os.environ.get('AIRIPC_WORKSPACE_ROOT','/home/user/airi')).resolve()
PORT = os.environ.get('AIRI_PORT','9010')

class MaintenanceManager:
    LEVELS={1:"retry",2:"component_restart",3:"component_repair",4:"runtime_rebuild",5:"escalation"}
    def _probe_url(self,url):
        try:
            with urllib.request.urlopen(url,timeout=3) as r: return {"ok":r.status==200}
        except Exception as e: return {"ok":False,"error":str(e)}
    def run(self):
        checks={}
        try:
            du=shutil.disk_usage(BASE.parent); checks['disk']={'ok':du.free>512*1024*1024,'free_gb':round(du.free/1e9,3),'used_percent':round(100*(du.total-du.free)/du.total,2)}
        except Exception as e: checks['disk']={'ok':False,'error':str(e)}
        try: checks['processes']={'ok':subprocess.run(['bash','-lc','pgrep -f "[u]vicorn.*--port ${AIRI_PORT:-9010}" >/dev/null'],capture_output=True).returncode==0}
        except Exception as e: checks['processes']={'ok':False,'error':str(e)}
        checks['display']={'ok':bool(os.environ.get('DISPLAY',':99')),'display':os.environ.get('DISPLAY',':99')}
        checks['mcp']=self._probe_url(f'http://127.0.0.1:{PORT}/status'); REGISTRY.record('mcp', checks['mcp'].get('ok',False), error=checks['mcp'].get('error',''))
        checks['ready']=self._probe_url(f'http://127.0.0.1:{PORT}/ready'); REGISTRY.record('ready', checks['ready'].get('ok',False), error=checks['ready'].get('error',''))
        overall=all(bool(v.get('ok')) for v in checks.values()); row={'timestamp':now(),'overall_ok':overall,'checks':checks,'recommended_level':1 if overall else 2,'reliability':REGISTRY.summary()}
        save_json('maintenance.json',row); return row
    def recover(self,level='auto',confirm=None):
        current=self.run(); lvl=2 if level in ('auto',None) and not current['overall_ok'] else int(level) if str(level).isdigit() else None
        if lvl is None: return {'ok':False,'error':'level must be auto or 1..5'}
        if current['overall_ok'] and lvl==1: return {'ok':True,'level':1,'action':'retry','result':current}
        record={'timestamp':now(),'requested_level':lvl,'current':current,'action':self.LEVELS[lvl]}
        try:
            if lvl==1:
                result=self.run(); record.update({'ok':result['overall_ok'],'result':result})
            elif lvl==2:
                log=str(BASE/'logs/maintenance-restart.log'); cmd=f'cd {BASE} && AIRI_FORCE_RESTART=1 sh computer/start.sh'
                with open(log,'ab') as f: subprocess.Popen(['/bin/bash','-lc',cmd],stdout=f,stderr=f,start_new_session=True)
                record.update({'ok':True,'started':True,'detail':'component restart delegated to start.sh','log':log})
            elif lvl==3:
                py=str(BASE/'.venv/bin/python');
                p=subprocess.run([py,'-m','pip','install','--disable-pip-version-check','-r',str(BASE/'computer/requirements.txt')],text=True,capture_output=True,timeout=120)
                record.update({'ok':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-3000:],'stderr':p.stderr[-3000:]})
            elif lvl==4 and confirm=='REBUILD':
                log=str(BASE/'logs/maintenance-rebuild.log'); f=open(log,'ab'); subprocess.Popen(['/bin/sh',str(BASE/'scripts/airi-rebuild')],stdout=f,stderr=f,start_new_session=True); record.update({'ok':True,'started':True,'log':log})
            elif lvl==4:
                record.update({'ok':False,'needs_confirmation':True,'required_confirmation':'REBUILD'})
            else:
                record.update({'ok':False,'escalation':{'reason':'automatic recovery exhausted or explicitly requested','recommendation':'inspect maintenance.json and logs'}})
        except Exception as exc: record.update({'ok':False,'error':str(exc)})
        save_json('maintenance-last-recovery.json',record); return record
    def history(self): return load_json('maintenance.json',{})
