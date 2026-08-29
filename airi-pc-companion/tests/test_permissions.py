from companion.permissions import authorize

def test_read_only(): assert authorize('screen','screenshot')[0]
def test_delete_requires_confirmation(): assert not authorize('delete','x','DESTRUCTIVE',False)[0]
def test_delete_confirmed(): assert authorize('delete','x','DESTRUCTIVE',True)[0]
