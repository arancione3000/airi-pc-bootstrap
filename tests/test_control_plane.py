from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "computer"))
from control_plane.capability_manager import CapabilityManager
from control_plane.transaction_engine import TransactionEngine

def test_capability_route():
    m=CapabilityManager(); m.discover(["computer_file_read","computer_terminal_run"])
    m.probe("computer_file_read", True, 10); m.probe("computer_terminal_run", False, 900, "timeout")
    assert m.route(["computer_file_read","computer_terminal_run"])["selected"] == "computer_file_read"

def test_transaction_undo():
    f=Path(".ai/control_plane/test-tx.txt"); f.parent.mkdir(parents=True,exist_ok=True); f.write_text("before")
    e=TransactionEngine(); tx=e.begin([".ai/control_plane/test-tx.txt"], "test"); f.write_text("after"); e.rollback(tx["id"])
    assert f.read_text()=="before"; f.unlink()

def test_task_resume_and_audit():
    from control_plane.task_engine import TaskEngine
    t=TaskEngine().start("demo",[{"id":"a","title":"analyze"},{"id":"b","title":"test","depends_on":["a"]}],["tests"])
    e=TaskEngine(); e.update("a","completed",output={"ok":True}); row=e.read(t["id"]); assert row["current"]=="b" and row["nodes"][1]["status"]=="running"

def test_audit_redacts_secrets():
    from control_plane.audit_engine import AuditEngine
    e=AuditEngine(); e.event(kind="secret-test", token="token=supersecret", note="authorization: Bearer-XYZ")
    row=e.tail(1)[-1]; raw=str(row)
    assert "supersecret" not in raw and "Bearer-XYZ" not in raw

def test_project_index_symbol_search():
    from control_plane.project_index import ProjectIndex
    idx=ProjectIndex(); summary=idx.refresh(["computer/control_plane"])
    assert summary["files"] >= 7
    assert any(x["symbol"]=="ControlPlane" for x in idx.search("ControlPlane"))

def test_skill_registry_and_checksum():
    from control_plane.skill_manager import SkillManager
    m=SkillManager(); data=m.refresh(); assert data["skills"]
    name=next(iter(data["skills"])); assert m.verify(name)["status"]=="valid"

def test_maintenance_health():
    from control_plane.maintenance import MaintenanceManager
    result=MaintenanceManager().run()
    assert result["overall_ok"] is True and result["checks"]["mcp"]["ok"] is True and result["checks"]["ready"]["ok"] is True

def test_recovery_level_four_requires_barrier():
    from control_plane.maintenance import MaintenanceManager
    result=MaintenanceManager().recover(4)
    assert result["ok"] is False and result["needs_confirmation"] is True and result["required_confirmation"]=="REBUILD"
