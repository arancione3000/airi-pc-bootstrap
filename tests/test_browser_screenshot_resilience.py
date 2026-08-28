from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "computer"))
import server

class DummyPage:
    def screenshot(self, *args, **kwargs):
        raise RuntimeError("Page.screenshot: Target crashed")

def test_browser_screenshot_falls_back_to_live_display(monkeypatch):
    class DummyBrowser:
        def _ensure(self): return DummyPage()
        def _close_worker(self): pass
        def call(self, fn, *args, **kwargs): return fn(*args, **kwargs)
    monkeypatch.setattr(server, "screenshot_image", lambda: server.Image.new("RGB", (1280, 800), "white"))
    result=server._BrowserManager.screenshot(DummyBrowser())
    assert result["ok"] is True
    assert result["method"] == "x11_fallback"
    assert result["width"] == 1280 and result["height"] == 800
    assert len(result["data_base64"]) > 100
