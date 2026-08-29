from __future__ import annotations
import time
class Timing:
    LIMITS={'instant':0.05,'short':0.25,'medium':1.0,'long':5.0}
    @classmethod
    def bounded_sleep(cls,kind): time.sleep(min(cls.LIMITS[kind],cls.LIMITS['long']))
