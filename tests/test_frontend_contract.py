from pathlib import Path


def test_frontend_stops_for_terminal_states_and_has_timeout():
    source = Path("app/static/app.js").read_text()
    assert '"failed", "stopped", "canceled"' in source
    assert "POLL_TIMEOUT_MS" in source
