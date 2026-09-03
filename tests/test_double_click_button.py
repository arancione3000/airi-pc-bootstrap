import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'computer'))
from server import _normalize_button
def test_normalize_button_accepts_pyautogui_names():
    assert _normalize_button('left')=='left'; assert _normalize_button('middle')=='middle'; assert _normalize_button('right')=='right'
def test_normalize_button_converts_legacy_numeric_codes():
    assert _normalize_button(0)=='left'; assert _normalize_button(1)=='middle'; assert _normalize_button(2)=='right'; assert _normalize_button('0')=='left'; assert _normalize_button('2')=='right'
def test_normalize_button_rejects_invalid_values():
    for value in (3,-1,True,'button',None):
        try: _normalize_button(value)
        except ValueError: pass
        else: raise AssertionError(f'invalid button accepted: {value!r}')
