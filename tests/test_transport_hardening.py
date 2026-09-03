from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_long_lived_startups_are_detached():
    start = read("computer/start.sh")
    session = read("scripts/airi-next-session")
    rebuild = read("scripts/airi-session-rebuild")
    watchdog = read("scripts/airi-watchdog")
    connect = read("scripts/airi-connect")
    for text in (start, session, rebuild, watchdog, connect):
        assert "nohup" in text
        assert "< /dev/null" in text


def test_pinggy_ssh_uses_failure_and_keepalive_options():
    text = read("scripts/airi-tunnel-supervisor")
    for token in ("ssh -N", "BatchMode=yes", "ExitOnForwardFailure=yes", "ServerAliveInterval=20", "ServerAliveCountMax=3", "TCPKeepAlive=yes", "ConnectTimeout=10", "< /dev/null"):
        assert token in text


def test_tailscale_is_authoritative_and_pinggy_is_fallback():
    text = read("scripts/airi-next-session")
    assert 'AIRI_ALLOW_PINGGY_FALLBACK:-1' in text
    tunnel = read("scripts/airi-tunnel-supervisor")
    assert "tailscale_healthy" in tunnel
    assert 'write_status "standby"' in tunnel
    assert 'atomic_write "$BASE/.ai/state/airi-endpoint.json"' in tunnel


def test_singleton_guards_exist_for_long_lived_transport_helpers():
    tailscale = read("scripts/airi-tailscale-supervisor")
    relay = read("scripts/airi-relay-updater")
    assert 'LOCKDIR="$STATE/supervisor.lock"' in tailscale
    assert 'mkdir "$LOCKDIR"' in tailscale
    assert 'LOCKDIR="$STATE/relay-updater.lock"' in relay
    assert 'mkdir "$LOCKDIR"' in relay


def test_auto_connect_preserves_next_failure_status():
    text = read("scripts/airi-auto-connect")
    assert 'else\n  A_STATUS=$?' in text


def test_relay_retries_failed_publish():
    text = read("scripts/airi-relay-updater")
    assert "if curl -fsS" in text
    assert 'LAST="$U"' in text
    assert 'then\n            LAST="$U"' in text
