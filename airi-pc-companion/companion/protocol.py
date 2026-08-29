from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import secrets, time

@dataclass(frozen=True)
class Request:
    request_id: str
    capability: str
    action: str
    arguments: dict[str, Any]
    expires_at: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> 'Request':
        rid=str(d.get('request_id') or secrets.token_urlsafe(12))
        capability=str(d.get('capability') or '')
        action=str(d.get('action') or '')
        args=d.get('arguments') or {}
        if not isinstance(args, dict): raise ValueError('arguments must be object')
        exp=float(d.get('expires_at') or (time.time()+15))
        if exp <= time.time(): raise ValueError('request expired')
        if len(rid)>128 or len(capability)>64 or len(action)>64: raise ValueError('field too long')
        return Request(rid, capability, action, args, exp)

    def to_dict(self) -> dict[str, Any]:
        return {'request_id':self.request_id,'capability':self.capability,'action':self.action,'arguments':self.arguments,'expires_at':self.expires_at}

def ok(req: Request, result: Any) -> dict[str, Any]:
    return {'request_id':req.request_id,'success':True,'result':result,'timestamp':datetime.now(timezone.utc).isoformat()}

def err(req: Request, error_class: str, message: str) -> dict[str, Any]:
    return {'request_id':req.request_id,'success':False,'error_class':error_class,'message':message[:500],'timestamp':datetime.now(timezone.utc).isoformat()}
