"""Tests for pinning the MCP bridge to one ChimeraX with CHIMERAX_REST_PORT.

The bridge's documented client configurations have always shown a
CHIMERAX_REST_HOST / CHIMERAX_REST_PORT env block, but nothing read it: the
host and port were module constants. Without it there is no way to tell one
bridge "talk to *that* ChimeraX", which is what headless and parallel callers
need -- they start their own REST servers and must not have the bridge wander
onto a sibling's instance or launch a GUI behind their back.

Pinning therefore implies three things, and each is tested here: the port is
used, discovery is skipped, and auto-start is refused.

The settings are read at import time, so these tests reload the module under a
patched environment. That is also the shape of the real contract -- an MCP
stdio server is spawned per session with its env already set.
"""

import asyncio
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("mcp")

from chimerax.mcpserver import chimerax_mcp_bridge as bridge  # noqa: E402


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


def _serve():
    """A stand-in ChimeraX REST server on an ephemeral port."""
    body = json.dumps(CHIMERAX_ENVELOPE).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/run"):
                payload, ctype = body, "application/json"
            else:
                payload, ctype = CMDLINE_HTML, "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def shutdown():
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    return httpd.server_address[1], shutdown


@pytest.fixture
def chimerax():
    port, shutdown = _serve()
    yield port
    shutdown()


@pytest.fixture
def pinned(monkeypatch):
    """Reload the bridge with CHIMERAX_REST_PORT set to a given port."""
    def _pin(port):
        monkeypatch.setenv("CHIMERAX_REST_PORT", str(port))
        return importlib.reload(bridge)
    yield _pin
    # Restore the unpinned module so later tests see the normal defaults.
    monkeypatch.delenv("CHIMERAX_REST_PORT", raising=False)
    importlib.reload(bridge)


@pytest.fixture(autouse=True)
def close_bridge_session():
    """Drop the cached aiohttp session; it is bound to a now-closed loop."""
    yield
    # reload() mutates the module in place, so this is the live object either way.
    session = bridge._session
    if session is not None and not session.closed:
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            session.close())
    bridge._session = None


def run(coro):
    return asyncio.run(coro)


# --- the pin is read at all -------------------------------------------------

def test_pin_sets_the_default_port(pinned):
    mod = pinned(9123)
    assert mod.PINNED_PORT == 9123
    assert mod.DEFAULT_CHIMERAX_PORT == 9123
    assert mod._default_port == 9123


def test_host_is_configurable(monkeypatch):
    monkeypatch.setenv("CHIMERAX_REST_HOST", "127.0.0.1")
    mod = importlib.reload(bridge)
    try:
        assert mod.CHIMERAX_HOST == "127.0.0.1"
    finally:
        monkeypatch.delenv("CHIMERAX_REST_HOST", raising=False)
        importlib.reload(bridge)


def test_unset_env_keeps_the_historical_defaults():
    assert bridge.PINNED_PORT is None
    assert bridge.DEFAULT_CHIMERAX_PORT == 8080
    assert bridge.CHIMERAX_HOST == "localhost"


@pytest.mark.parametrize("value", ["", "http://localhost:8080", "eighty-eighty",
                                   "0", "-1", "65536"])
def test_unusable_values_fall_back_rather_than_crash(monkeypatch, value):
    # An exception here would be an opaque startup failure in a stdio server.
    monkeypatch.setenv("CHIMERAX_REST_PORT", value)
    mod = importlib.reload(bridge)
    try:
        assert mod.PINNED_PORT is None
        assert mod.DEFAULT_CHIMERAX_PORT == 8080
    finally:
        monkeypatch.delenv("CHIMERAX_REST_PORT", raising=False)
        importlib.reload(bridge)


# --- pinning skips discovery ------------------------------------------------

def test_pinned_lookup_skips_the_scan(pinned):
    mod = pinned(9123)
    # No server is listening on 9123. Unpinned, this would scan the fallback
    # list; pinned, it must return the pinned port regardless -- a busy or
    # briefly-down ChimeraX must not divert commands to another instance.
    assert run(mod.find_best_chimerax_instance()) == 9123


def test_pinned_candidate_list_is_empty(pinned):
    mod = pinned(9123)
    mod._instances[9200] = {"status": "running"}
    try:
        assert mod._candidate_ports() == []
    finally:
        mod._instances.clear()


def test_unpinned_lookup_still_scans():
    # Guard the default path: the fallback list is still consulted.
    assert bridge._candidate_ports()


# --- pinning refuses to auto-start ------------------------------------------

def test_pinned_start_refuses_when_nothing_answers(pinned):
    mod = pinned(9123)
    ok, port = run(mod.start_chimerax())
    assert ok is False
    assert port == 9123


def test_pinned_start_never_launches_a_process(pinned, monkeypatch):
    mod = pinned(9123)
    monkeypatch.setattr(mod, "find_chimerax_executable",
                        lambda: pytest.fail("must not look for an executable"))
    monkeypatch.setattr(mod, "start_chimerax_daemon",
                        lambda port: pytest.fail("must not launch ChimeraX"))
    assert run(mod.start_chimerax())[0] is False


def test_pinned_start_succeeds_when_the_pinned_server_answers(pinned, chimerax):
    mod = pinned(chimerax)
    assert run(mod.start_chimerax()) == (True, chimerax)


# --- the pin holds against the tools ----------------------------------------

def test_set_default_session_is_refused_while_pinned(pinned, chimerax):
    mod = pinned(9123)
    message = run(mod.set_default_session(chimerax))
    assert "pinned" in message.lower()
    assert mod._default_port == 9123


def test_start_new_session_is_refused_while_pinned(pinned, monkeypatch):
    mod = pinned(9123)
    monkeypatch.setattr(mod, "find_chimerax_executable",
                        lambda: pytest.fail("must not look for an executable"))
    message = run(mod.start_new_chimerax_session())
    assert "pinned" in message.lower()
    assert "9123" in message


def test_set_default_session_still_works_unpinned(chimerax):
    saved = bridge._default_port
    try:
        run(bridge.set_default_session(chimerax))
        assert bridge._default_port == chimerax
    finally:
        bridge._default_port = saved
