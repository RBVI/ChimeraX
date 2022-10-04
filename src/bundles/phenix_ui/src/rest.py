# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# ---------------------------------------------------------------------------------------
#
def phenix_rest_server(session, phenix_location):
    if not hasattr(session, '_phenix_rest_server'):
        session._phenix_rest_server = PhenixRestServer(session, phenix_location)
    return session._phenix_rest_server

# ---------------------------------------------------------------------------------------
#
class PhenixRestServer:
    def __init__(self, session, phenix_location):
        from .locate import find_phenix_command
        exe_path = find_phenix_command(session, 'phenix.rest_server', phenix_location)
        from subprocess import Popen, PIPE
        # Use start_new_session = True so server is killed when ChimeraX exits
        self._server_process = Popen([exe_path], stdout = PIPE, stdin = PIPE,
                                     start_new_session = True)
        # Need to kill server when ChimeraX exits otherwise it lives on.
        import atexit
        atexit.register(self._terminate)
        self._url = 'http://localhost:8000/'
        self._wait_for_startup()

    def _terminate(self):
        p = self._server_process
#        p.terminate()  # Does not kill worker children
#        from signal import SIGHUP
#        p.send_signal(SIGHUP)  # Does not kill worker children
        # Have to kill process group so workers exit
        import os, signal
        pgrp = os.getpgid(p.pid)
        os.killpg(pgrp, signal.SIGINT)
        p.wait()

    def _wait_for_startup(self):
        p = self._server_process
        for line in p.stdout:
            if line.find(b'Started Flask app') != -1:
                break

    def start_job(self, program, args):
        params = {'program': program, 'args': args}
        import requests
        output = requests.get(url=self._url + 'start_job', params=params)
        job_id = output.json()['job_id']
        job = PhenixRestJob(job_id, self)
        return job

    def wait(self, job_id, check_interval = 1):
        while True:
            import requests
            output = requests.get(url=self._url + 'job_status', params={'job_id': job_id})
            if output.json()['status'] == 'complete':
                break
            import time
            time.sleep(check_interval)

    def output(self, job_id):
        import requests
        result = requests.get(url=self._url + 'get_job_result', params={'job_id': job_id})
        result = result.json()
        output = result['result'] # TODO: Remove this after Feb 10 Phenix.
        return output
        
# ---------------------------------------------------------------------------------------
#
class PhenixRestJob:
    def __init__(self, job_id, rest_server):
        self._job_id = job_id
        self._rest_server = rest_server
        self._output = None

    def wait(self, check_interval = 1):
        self._rest_server.wait(self._job_id, check_interval=check_interval)

    @property
    def result(self):
        result = eval(self.output['result'])  # TODO: No eval after Feb 10 Phenix
        return result

    @property
    def stdout(self):
        self._output['stdout']

    @property
    def stderr(self):
        self._output['stderr']

    @property
    def output(self):
        if self._output is None:
            self._output = self._rest_server.output(self._job_id)
        return self._output
