"""Tests for the MCP bridge's handling of a contended REST port.

Port 8080 is the bridge's default and one of the most contended ports in
software. If an unrelated HTTP server holds it, the bridge must notice rather
than mistake it for ChimeraX -- otherwise every command "succeeds" while doing
nothing, which is exactly what a stray mock server answering `{}` to every
request once produced.

These are pure unit tests: they stand up small HTTP stubs on ephemeral ports
and never need a ChimeraX session.
"""

import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("mcp")

from chimerax.mcpserver import chimerax_mcp_bridge as bridge  # noqa: E402


# The real page the bridge fingerprints ChimeraX with; only the markers matter.
CMDLINE_HTML = b"""<html lang="en"><!--
=== UCSF ChimeraX Copyright ===
--><head><meta charset="utf-8">
<title>Chimera Web Command Line</title>
</head><body></body></html>
"""

CHIMERAX_ENVELOPE = {
    "json values": [None],
    "python values": ["opened 1 model"],
    "log messages": {"info": ["1 model opened"]},
    "error": None,
}


def _make_handler(run_body, run_content_type, serve_cmdline_html):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/run"):
                body, ctype = run_body, run_content_type
            elif serve_cmdline_html:
                body, ctype = CMDLINE_HTML, "text/html"
            else:
                # Models a server that answers everything identically -- the
                # behaviour that fooled the old status-code-only probe.
                body, ctype = run_body, run_content_type
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def _serve(handler_cls):
    """Run handler_cls on an ephemeral port; return (port, shutdown_callable)."""
    httpd = HTTPServer(("localhost", 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def shutdown():
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    return httpd.server_address[1], shutdown


@pytest.fixture
def imposter():
    """A server that answers 200 + `{}` to every path, ChimeraX or not."""
    port, shutdown = _serve(_make_handler(b"{}", "application/json", False))
    yield port
    shutdown()


@pytest.fixture
def chimerax_json_mode():
    """A stand-in for a real ChimeraX REST server started with `json true`."""
    body = json.dumps(CHIMERAX_ENVELOPE).encode()
    port, shutdown = _serve(_make_handler(body, "application/json", True))
    yield port
    shutdown()


@pytest.fixture
def chimerax_no_json_mode():
    """Real ChimeraX, but `remotecontrol rest start` without `json true`."""
    port, shutdown = _serve(
        _make_handler(b"1 model opened", "text/plain", True))
    yield port
    shutdown()


@pytest.fixture(autouse=True)
def clean_bridge_state():
    """Keep the bridge's module-level instance registry out of other tests."""
    saved = dict(bridge._instances)
    bridge._instances.clear()
    yield
    bridge._instances.clear()
    bridge._instances.update(saved)


@pytest.fixture(autouse=True)
def close_bridge_session():
    yield
    session = bridge._session
    if session is not None and not session.closed:
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            session.close())
    bridge._session = None


def run(coro):
    return asyncio.run(coro)


# --- the liveness probe must verify identity, not just liveness -------------

def test_probe_rejects_imposter(imposter):
    assert run(bridge.is_chimerax_running(imposter)) is False


def test_probe_accepts_real_chimerax(chimerax_json_mode):
    assert run(bridge.is_chimerax_running(chimerax_json_mode)) is True


def test_probe_accepts_chimerax_without_json_mode(chimerax_no_json_mode):
    # cmdline.html is served regardless of the REST server's json setting, so
    # this is still a ChimeraX -- the distinction is drawn later, on /run.
    assert run(bridge.is_chimerax_running(chimerax_no_json_mode)) is True


def test_probe_rejects_dead_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        dead_port = s.getsockname()[1]
    assert run(bridge.is_chimerax_running(dead_port)) is False


# --- an imposter must never be mistaken for a successful command ------------

def test_imposter_does_not_yield_a_fake_success(imposter, monkeypatch):
    """The regression: `{}` used to format as "Command completed successfully"."""
    monkeypatch.setattr(
        bridge, "start_chimerax",
        lambda *a, **kw: _async_return((False, imposter)))

    with pytest.raises(Exception) as excinfo:
        run(bridge.run_chimerax_command("open 1gcn", imposter))

    assert "Command completed successfully" not in str(excinfo.value)
    assert str(imposter) in str(excinfo.value)


def test_imposter_triggers_autostart_elsewhere(imposter, chimerax_json_mode,
                                               monkeypatch):
    """Squatted port -> start ChimeraX on another port -> retry -> real result."""
    started = {}

    async def fake_start(port=None, session_name=None, force_new=False):
        started["asked_for"] = port
        return True, chimerax_json_mode

    monkeypatch.setattr(bridge, "start_chimerax", fake_start)

    result = run(bridge.run_chimerax_command("open 1gcn", imposter))

    assert started["asked_for"] == imposter
    assert result["return_values"] == ["opened 1 model"]
    assert result["logs"] == {"info": ["1 model opened"]}


def test_real_chimerax_without_json_mode_is_not_duplicated(
        chimerax_no_json_mode, monkeypatch):
    """Don't start a second ChimeraX just because JSON mode is off."""
    def refuse(*args, **kwargs):
        raise AssertionError("start_chimerax must not be called here")

    monkeypatch.setattr(bridge, "start_chimerax", refuse)

    with pytest.raises(Exception) as excinfo:
        run(bridge.run_chimerax_command("open 1gcn", chimerax_no_json_mode))

    assert "mcp start" in str(excinfo.value)


def test_healthy_chimerax_still_works(chimerax_json_mode):
    result = run(bridge.run_chimerax_command("open 1gcn", chimerax_json_mode))
    assert result["return_values"] == ["opened 1 model"]


# --- the default port must not be exempt from the availability check --------

def test_start_chimerax_moves_off_a_squatted_default_port(imposter, monkeypatch):
    """The guard `if port != _default_port` used to skip this check entirely."""
    launched = {}

    def fake_daemon(port):
        launched["port"] = port
        return True

    monkeypatch.setattr(bridge, "_default_port", imposter)
    monkeypatch.setattr(bridge, "start_chimerax_daemon", fake_daemon)
    monkeypatch.setattr(bridge, "find_chimerax_executable", lambda: "/bin/true")
    # Skip the "is an existing REST server already up?" sweep and the 30 s wait.
    monkeypatch.setattr(bridge, "check_existing_rest_server",
                        lambda: _async_return((False, imposter)))
    monkeypatch.setattr(bridge, "is_chimerax_running",
                        lambda port=None: _async_return(port == launched.get("port")))

    success, port = run(bridge.start_chimerax(imposter))

    assert success
    assert port != imposter, "should have moved off the squatted port"
    assert launched["port"] == port


# --- discovery must not forget an instance started on an unusual port -------

def test_candidate_ports_includes_started_instances(monkeypatch):
    monkeypatch.setattr(bridge, "_default_port", 8080)
    bridge._instances[9137] = {"status": "running"}

    ports = bridge._candidate_ports()

    assert ports[0] == 9137, "bridge-started instances should be probed first"
    assert 8080 not in ports, "the default port is checked separately"
    assert len(ports) == len(set(ports)), "no duplicate probes"


def test_candidate_ports_drops_default_from_fallbacks(monkeypatch):
    monkeypatch.setattr(bridge, "_default_port", 8082)
    assert 8082 not in bridge._candidate_ports()


async def _async_return_impl(value):
    return value


def _async_return(value):
    return _async_return_impl(value)
