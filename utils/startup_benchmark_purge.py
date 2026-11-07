"""Authorize macOS cache purges once for one bounded benchmark series.

The privileged process executes fixed, inline code through Apple's admin
prompt. It only runs /usr/sbin/purge, with a fixed request quota and idle
expiry. Closing the private socket ends the helper; no password or persistent
sudo configuration is stored. The benchmark itself remains unprivileged.

Usage: ``with PurgeSeries(number_of_launches) as series:`` and call
``series.purge()`` before each cold launch. A failed or expired helper stops
that series rather than opening another authorization dialog.
"""
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import tempfile
import time


_SERVER = r'''
import socket
import subprocess
import sys


def serve(connection, remaining):
    connection.settimeout(120)
    with connection.makefile('rwb', buffering=0) as stream:
        for _ in range(remaining):
            request = stream.readline(16)
            if request in (b'', b'QUIT\n'):
                return
            if request != b'PURGE\n':
                stream.write(b'ERROR invalid request\n')
                return
            try:
                result = subprocess.run(['/usr/sbin/purge'], capture_output=True,
                    timeout=45)
            except subprocess.TimeoutExpired:
                stream.write(b'ERROR purge timed out\n')
                return
            if result.returncode:
                stream.write(b'ERROR purge failed\n')
                return
            stream.write(b'OK\n')


if __name__ == '__main__':
    path, token, quota = sys.argv[1:]
    quota = int(quota)
    if not 1 <= quota <= 1000:
        raise ValueError('Invalid purge quota')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(15)
        connection.connect(path)
        connection.sendall(('READY ' + token + '\n').encode('ascii'))
        serve(connection, quota)
'''


def authorization_script(path, token, quota):
    # shlex quotes shell arguments; AppleScript quoting is a separate layer.
    # Keep the inline Python argument on one line to avoid AppleScript escape
    # ambiguities. Never execute a mutable script file with root privileges.
    command = shlex.join(['/usr/bin/python3', '-I', '-c',
        'exec(' + repr(_SERVER) + ')', str(path), token, str(quota)])
    literal = '"' + command.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return 'do shell script ' + literal + ' with administrator privileges'


class PurgeSeries:
    """One administrator authorization, reused for at most max_purges calls."""

    def __init__(self, max_purges):
        if type(max_purges) is not int or not 1 <= max_purges <= 1000:
            raise ValueError('A purge series must contain 1 to 1000 purges')
        self.remaining = max_purges
        self._quota = max_purges
        self._directory = self._listener = self._connection = self._stream = None
        self._process = None
        self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exception):
        self.close()

    def start(self):
        if self._started:
            raise RuntimeError('A purge series cannot be authorized twice')
        self._started = True
        try:
            self._directory = tempfile.TemporaryDirectory(prefix='chimerax-purge-',
                dir='/private/tmp')
            path = Path(self._directory.name) / 'control.sock'
            self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener.bind(str(path))
            os.chmod(path, 0o600)
            self._listener.listen(1)
            self._listener.settimeout(0.25)
            token = secrets.token_hex(24)
            self._process = subprocess.Popen(['/usr/bin/osascript', '-e',
                authorization_script(path, token, self._quota)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.monotonic() + 120
            while True:
                if self._process.poll() is not None:
                    _, error = self._process.communicate()
                    raise RuntimeError('Cache-purge authorization failed: '
                        + error.decode(errors='replace').strip())
                if time.monotonic() >= deadline:
                    raise TimeoutError('Cache-purge authorization timed out')
                try:
                    self._connection, _ = self._listener.accept()
                    break
                except socket.timeout:
                    continue
            self._connection.settimeout(50)
            self._stream = self._connection.makefile('rwb', buffering=0)
            ready = self._stream.readline(128)
            if ready != ('READY ' + token + '\n').encode('ascii'):
                raise RuntimeError('Unexpected cache-purge helper connection')
            self._listener.close()
            self._listener = None
            return self
        except BaseException:
            self.close()
            raise

    def purge(self):
        if self._stream is None or self.remaining <= 0:
            raise RuntimeError('Cache-purge series is closed or exhausted')
        try:
            self._stream.write(b'PURGE\n')
            reply = self._stream.readline(128)
            if reply != b'OK\n':
                raise RuntimeError(reply.decode(errors='replace').strip()
                    or 'Cache-purge helper disconnected')
            self.remaining -= 1
        except BaseException:
            # Never silently launch another authorization prompt mid-series.
            self.close()
            raise

    def close(self):
        self.remaining = 0
        if self._stream is not None:
            try:
                self._stream.write(b'QUIT\n')
            except OSError:
                pass
            try:
                self._stream.close()
            except OSError:
                pass
            self._stream = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self._process is not None:
            try:
                self._process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                # Closing the socket makes the bounded helper exit after any
                # in-flight purge finishes; terminate the unprivileged wrapper.
                try:
                    self._process.terminate()
                    self._process.communicate(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._process = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
