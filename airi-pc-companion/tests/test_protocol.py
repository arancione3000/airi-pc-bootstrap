import time
from companion.protocol import Request

def test_request_validation():
 r=Request.from_dict({'capability':'system','action':'info','arguments':{},'expires_at':time.time()+5}); assert r.capability=='system'

def test_expired_rejected():
 try: Request.from_dict({'capability':'system','action':'info','expires_at':time.time()-1})
 except ValueError: pass
 else: assert False
