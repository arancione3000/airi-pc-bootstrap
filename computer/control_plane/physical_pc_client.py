from __future__ import annotations
import json, os, time, urllib.request, urllib.error
from dataclasses import dataclass

@dataclass
class PhysicalPCClient:
    base_url: str
    token: str
    timeout: float = 10.0

    @classmethod
    def from_env(cls):
        return cls(os.environ.get('AIRIPC_COMPANION_URL','').rstrip('/'), os.environ.get('AIRIPC_COMPANION_TOKEN',''))

    def _request(self, method, path, payload=None):
        if not self.base_url or not self.token: raise RuntimeError('PHYSICAL_PC_NOT_CONFIGURED')
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(self.base_url+path,data=data,method=method,headers={'Authorization':f'Bearer {self.token}','Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as r: return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return {'success':False,'error_class':'HTTP_ERROR','status':e.code}
        except Exception as e:
            return {'success':False,'error_class':type(e).__name__}

    def health(self):
        if not self.base_url: return {'online':False,'configured':False}
        try:
            with urllib.request.urlopen(self.base_url+'/health',timeout=self.timeout) as r: return json.loads(r.read().decode())
        except Exception as e: return {'online':False,'error_class':type(e).__name__}

    def status(self): return self._request('GET','/status')
    def execute(self, capability, action, arguments=None, confirmed=False, ttl=15):
        now=time.time(); return self._request('POST','/v1/execute',{'request_id':os.urandom(8).hex(),'capability':capability,'action':action,'arguments':arguments or {},'expires_at':now+ttl,'meta':{'confirmed':bool(confirmed)}})
