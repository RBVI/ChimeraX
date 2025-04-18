# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def boltz_install(session, directory, download_model_weights_and_ccd = True):
    # Check that directory either does not exist or is empty.
    from os.path import exists, isdir
    if exists(directory):
        from os import listdir
        if not isdir(directory) or len(listdir(directory)):
            from chimerax.core.errors import UserError
            raise UserError('You must install Boltz into a new or empty directory')

    ib = InstallBoltz(session, directory, download_model_weights_and_ccd)
    return ib
            
# ------------------------------------------------------------------------------
#
class InstallBoltz:

    def __init__(self, session, directory, download_model_weights_and_ccd = True):

        self._session = session
        self._directory = directory
        self._download_model_weights_and_ccd = download_model_weights_and_ccd
        self.finished_callback = None
        self.success = None

        if _create_boltz_virtual_environment(directory):
            self._pip_install_boltz(directory)
        else:
            success = False

    # ------------------------------------------------------------------------------
    #
    def _create_boltz_virtual_environment(self, directory):
        # Create Python virtual environment using ChimeraX Python as base for installing Boltz.
        from chimerax.core.python_utils import chimerax_python_executable
        python_exe = chimerax_python_executable()
        command = [python_exe, '-m', 'venv', directory]
        from subprocess import run
        p = run(command, capture_output = True)

        # Report success or failure of creating virtual environment.
        logger = self._session.logger
        if p.returncode == 0:
            logger.info(f'Successfully created Boltz Python virtual environment {directory}.')
        else:
            cmd = ' '.join(command)
            logger.error('Creating Python virtual environment failed.'
                         f'\nCommand: {cmd}'
                         f'\nstdout: {p.stdout}'
                         f'\nstderr: {p.stderr}')
            return False
        return True

    # ------------------------------------------------------------------------------
    #
    def _pip_install_boltz(self, directory):
        # Run pip install of Boltz in virtual environment
        logger = self._session.logger
        logger.info('Now installing Boltz and required packages from PyPi.  This may take tens of of minutes'
                    ' since Boltz uses many other packages totaling about 1 Gbyte of disk'
                    ' space including torch, scipy, rdkit, llvmlite, sympy, pandas, numpy, wandb, numba...')
        from os.path import join
        venv_python_exe = join(directory, 'bin', 'python')
        command = [venv_python_exe, '-m', 'pip', 'install', 'boltz']
        from subprocess import Popen, PIPE, STDOUT
        p = Popen(command, stdout = PIPE, stderr = STDOUT)

        logger.info(' '.join(command))

        # Echo subprocess output to the ChimeraX Log.
        log_subprocess_output(self._session, p, self._finished_pip_install)
        
    # ------------------------------------------------------------------------------
    #
    def _finished_pip_install(self, popen):
        # Report success of failure of pip install of Boltz
        popen.wait()  # Set returncode
        logger = self._session.logger
        if popen.returncode == 0:
            logger.info('Successfully installed Boltz.')
        else:
            logger.error('Boltz installation failed.  See ChimeraX Log for details.')
            self._finished(success = False)
            return

        # Remember the Boltz install directory
        from .settings import _boltz_settings
        settings = _boltz_settings(self._session)
        settings.boltz_install_location = self._directory
        settings.save()

        # Make the Boltz GUI show the install location.
        from .boltz_gui import boltz_panel
        p = boltz_panel(self._session)
        if p:
            p._install_directory.value = self._directory

        if self._download_model_weights_and_ccd:
            self._download_model_weights_and_ccd_database()
        else:
            self._finished()

    # ------------------------------------------------------------------------------
    #
    def _download_model_weights_and_ccd_database(self):
        from os.path import join, expanduser, exists
        venv_python_exe = join(directory, 'bin', 'python')

        self._data_path = expanduser('~/.boltz')
        if not exists(data_path):
            from os import mkdir
            mkdir(data_path)

        logger = self._session.logger
        logger.info('Downloading Boltz model parameters (3.3 GB) and chemical component database (330 MB)' +
                    f' to {data_path}')

        python_call = f'from boltz.main import download ; from pathlib import Path ; download(Path("{data_path}"))'
        command = [venv_python_exe, '-c', python_call]
        env = {}
        from sys import platform
        if platform == 'darwin':
            # On Mac the huggingface.co URLs get SSL certificate errors unless we setup certifi root certificates.
            import certifi
            env["SSL_CERT_FILE"] = certifi.where()
        from subprocess import Popen, PIPE, STDOUT
        p = Popen(command, stdout = PIPE, stderr = STDOUT, env = env)

        logger.info(' '.join(command))

        # Echo subprocess output to the ChimeraX Log.
        log_subprocess_output(session, p, self._finished_download_weights_and_ccd)

    # ------------------------------------------------------------------------------
    #
    def _finished_download_weights_and_ccd(session, popen, data_path):
        # Report success of failure downloading boltz model parameters and ccd
        popen.wait()  # Set returncode
        logger = self._session.logger
        success = (popen.returncode == 0)
        if success:
            logger.info(f'Boltz model parameters and CCD database are in {self._data_path}')
            success = self._make_ccd_atom_counts_file()
        else:
            logger.error(f'Downloading Boltz model parameters and CCD database to {self._data_path} failed.')

        self._finished(success)

    # ------------------------------------------------------------------------------
    #
    def _make_ccd_atom_counts_file(self):
        from os.path import join
        venv_python_exe = join(self._directory, 'bin', 'python')
        make_counts_path = join(dirname(__file__), 'make_ccd_atom_counts_file.py')
        command = [venv_python_exe, make_counts_path]
        from subprocess import run
        p = run(command, capture_output = True)

        # Report success or failure of creating virtual environment.
        logger = self._session.logger
        if p.returncode == 0:
            logger.info(f'Successfully created CCD atom counts file.')
        else:
            cmd = ' '.join(command)
            logger.error('Creating CCD atom counts file failed.')
                         f'\nCommand: {cmd}'
                         f'\nstdout: {p.stdout}'
                         f'\nstderr: {p.stderr}')
            return False
        return True

    # ------------------------------------------------------------------------------
    #
    def _finished(self, success = True):
        self.success = success
        if self.finished_callback:
            self.finished_callback(success)

# ------------------------------------------------------------------------------
#
class log_subprocess_output:
    def __init__(self, session, popen, finished_callback):
        self._session = session
        self._popen = popen
        self._finished_callback = finished_callback
        from queue import Queue
        self._queue = Queue()
        session.triggers.add_handler('new frame', self._log_queued_lines)
        from threading import Thread
        # Set daemon true so that ChimeraX exit is not blocked by the thread still running.
        self._thread = t = Thread(target = self._queue_output_in_thread, daemon = True)
        t.start()

    def _queue_output_in_thread(self):
        while True:
            line = self._popen.stdout.readline() # blocking read
            if not line:
                break
            self._queue.put(line)

    def _log_queued_lines(self, tname, tdata):
        while not self._queue.empty():
            line = self._queue.get()
            self._session.logger.info(line.decode('utf-8'))
        if not self._thread.is_alive():
            self._finished_callback()
            return 'delete handler'
            
# ------------------------------------------------------------------------------
#
def register_boltz_install_command(logger):
    from chimerax.core.commands import CmdDesc, register, SaveFolderNameArg, BoolArg
    desc = CmdDesc(
        required = [('directory', SaveFolderNameArg)],
        keyword = [('download_model_weights_and_ccd', BoolArg)],
        synopsis = 'Install Boltz from PyPi in a virtual environment'
    )
    register('boltz install', desc, boltz_install, logger=logger)

