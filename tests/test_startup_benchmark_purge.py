"""Exercise the purge protocol with an unprivileged, in-memory command runner."""
import importlib.util
import json
from pathlib import Path
import shlex
import socket
import subprocess
from threading import Thread
from types import SimpleNamespace

import pytest


_spec = importlib.util.spec_from_file_location('benchmark_purge',
    Path(__file__).parents[1] / 'utils' / 'startup_benchmark_purge.py')
purge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(purge)


def server_namespace(run):
    namespace = {'__name__': 'purge_protocol'}
    exec(purge._SERVER, namespace)
    namespace['subprocess'] = SimpleNamespace(run=run,
        TimeoutExpired=subprocess.TimeoutExpired)
    return namespace


@pytest.fixture
def helper(monkeypatch):
    calls, authorizations, errors = [], [], []
    state = SimpleNamespace(fail=False)
    temporary_directory = purge.tempfile.TemporaryDirectory
    monkeypatch.setattr(purge.tempfile, 'TemporaryDirectory',
        lambda **kwargs: temporary_directory(prefix=kwargs['prefix']))
    request = {}

    def script(path, token, quota):
        request.update(path=path, token=token, quota=quota)
        return 'mock authorization'

    def run(command, **kwargs):
        assert command == ['/usr/sbin/purge']
        assert kwargs == {'capture_output': True, 'timeout': 45}
        calls.append(command)
        return SimpleNamespace(returncode=int(state.fail))

    def launch(command, **kwargs):
        assert command == ['/usr/bin/osascript', '-e', 'mock authorization']
        authorizations.append(command)
        namespace = server_namespace(run)

        def serve():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(str(request['path']))
                    connection.sendall(('READY ' + request['token'] + '\n').encode())
                    namespace['serve'](connection, request['quota'])
            except Exception as error:
                errors.append(error)

        thread = Thread(target=serve)
        thread.start()

        def communicate(timeout=None):
            thread.join(timeout)
            if thread.is_alive():
                raise subprocess.TimeoutExpired(command, timeout)
            return b'', b''

        return SimpleNamespace(poll=lambda: None if thread.is_alive() else 0,
            communicate=communicate, terminate=lambda: None)

    monkeypatch.setattr(purge, 'authorization_script', script)
    monkeypatch.setattr(purge.subprocess, 'Popen', launch)
    return SimpleNamespace(calls=calls, authorizations=authorizations,
        errors=errors, request=request, state=state)


def test_one_authorization_covers_whole_series(helper):
    series = purge.PurgeSeries(3).start()
    directory = Path(series._directory.name)
    try:
        for _ in range(3):
            series.purge()
        assert len(helper.authorizations) == 1
        assert len(helper.calls) == 3
        with pytest.raises(RuntimeError, match='exhausted'):
            series.purge()
    finally:
        series.close()
    assert not directory.exists()
    assert not helper.errors
    with pytest.raises(RuntimeError, match='twice'):
        series.start()


def test_failure_closes_session_without_another_prompt(helper):
    series = purge.PurgeSeries(2).start()
    directory = Path(series._directory.name)
    helper.state.fail = True
    with pytest.raises(RuntimeError, match='purge failed'):
        series.purge()
    assert not directory.exists()
    assert len(helper.authorizations) == 1
    with pytest.raises(RuntimeError, match='closed'):
        series.purge()
    assert not helper.errors


def test_socket_disconnect_ends_helper_without_purging(helper):
    series = purge.PurgeSeries(2).start()
    series.close()
    assert not helper.calls
    assert len(helper.authorizations) == 1
    assert not helper.errors


def test_protocol_rejects_commands_instead_of_executing_them():
    calls = []
    namespace = server_namespace(lambda *args, **kwargs: calls.append(args))
    client, server = socket.socketpair()

    def serve():
        with server:
            namespace['serve'](server, 2)

    worker = Thread(target=serve)
    worker.start()
    with client:
        client.settimeout(2)
        client.sendall(b'PURGE;whoami\n')
        assert client.recv(128) == b'ERROR invalid request\n'
    worker.join(2)
    assert not worker.is_alive()
    assert calls == []


def test_shell_and_applescript_quoting_keep_arguments_literal():
    path = "/private/tmp/quote' and $(touch nope)/control.sock"
    script = purge.authorization_script(path, 'token', 4)
    literal = script.removeprefix('do shell script ').removesuffix(
        ' with administrator privileges')
    command = shlex.split(json.loads(literal))
    assert command[:3] == ['/usr/bin/python3', '-I', '-c']
    assert command[4:] == [path, 'token', '4']
    assert 'exec(' in command[3]


@pytest.mark.parametrize('quota', [0, -1, 1001, True, '4'])
def test_invalid_series_never_authorizes(quota):
    with pytest.raises(ValueError):
        purge.PurgeSeries(quota)
