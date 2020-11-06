# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
class SSHRemoteTunnel:
    '''
    Run ssh to set up a tunnel to a remote host.
    If the host just does not accept ssh connections it typically will take 60 seconds
    or more to time out.  This routine will return the process object and the client
    will have to call its poll() method after a minute or two to know if it failed.
    '''
    def __init__(self, remote, key_path, remote_port_range, local_port, log = None,
                 connection_timeout = 5, exit_check_interval = 1.0):
        host = remote.split('@')[1] if '@' in remote else remote
        self.host = host
        # TODO: If first port tried fails, try other ports.
        port_min, port_max = remote_port_range
        from random import randint
        remote_port = randint(port_min, port_max)
        self.remote_port = remote_port
        self.local_port = local_port
        self._log = log

        import sys
        if sys.platform == 'win32':
            ssh_exe = 'C:\\Windows\\System32\\OpenSSH\\ssh.exe'
            # Prevent console window from showing.
            from subprocess import STARTUPINFO, STARTF_USESHOWWINDOW
            startupinfo = STARTUPINFO()
            startupinfo.dwFlags |= STARTF_USESHOWWINDOW
        else:
            ssh_exe = 'ssh'
            startupinfo = None
        command = [
            ssh_exe,
            '-N',					# Do not execute remote command
            '-i', key_path,				# Private key for authentication
            # '-f',  # Test: Fork and exit when tunnel created, leaves no way to kill process
            '-o', 'ExitOnForwardFailure=yes',		# If remote port in use already exit instead of warn
            '-o', 'ConnectTimeout=%d' % connection_timeout, # Fail faster than 75 second TCP timeout if can't connect
            '-o', 'StrictHostKeyChecking=no',		# Don't ask about host authenticity on first connection
            '-R', '%d:localhost:%d' % (remote_port,local_port),	# Remote port forwarding
            remote,	# Remote machine
        ]
        from subprocess import Popen, PIPE
        try:
            p = Popen(command, stdout=PIPE, stderr=PIPE, startupinfo=startupinfo)
        except Exception as e:
            log.warning('meeting: failed to run "%s"\n%s' % (str(command), str(e)))
            return None

        self._popen = p
                 
        # Assure at exit that the process is terminated.
        import atexit
        atexit.register(self.close)

        # Periodically check if ssh has exited.
        if exit_check_interval is None:
            t = None
        else:
            t = _periodic_callback(exit_check_interval, self._check_for_process_exit)
        self._exit_check_timer = t

    # -----------------------------------------------------------------------------
    #
    def close(self):
        p = self._popen
        if p and p.poll() is None:
            p.terminate()
        self._popen = None

        ect = self._exit_check_timer
        if ect is not None:
            ect.stop()
            self._exit_check_timer = None

    # -----------------------------------------------------------------------------
    #
    def _check_for_process_exit(self):
        p = self._popen
        exit_code = p.poll()
        if exit_code is None:
            return	# Still running

        if exit_code == 0:
            self._log.info('meeting: ssh tunnel closed.')
        else:
            command = ' '.join(p.args)
            msg = 'meeting: ssh tunnel setup failed %s, exit code %d' % (command, exit_code)
            out = p.stdout.read().decode('utf-8')
            err = p.stderr.read().decode('utf-8')
            if out:
                msg += '\nssh stdout:\n%s' % out
            if err:
                msg += '\nssh stderr:\n%s' % err
            self._log.warning(msg)

        self._exit_check_timer.stop()

# -----------------------------------------------------------------------------
#
def _periodic_callback(interval, callback, *args, **kw):
    from PyQt5.QtCore import QTimer
    t = QTimer()
    def cb(callback=callback, args=args, kw=kw):
        callback(*args, **kw)
    t.timeout.connect(cb)
    t.setSingleShot(False)
    t.start(int(1000*interval))
    return t
