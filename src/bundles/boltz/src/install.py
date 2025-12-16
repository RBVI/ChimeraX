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

def boltz_install(session, directory = None, download_model_weights_and_ccd = True,
                  branch = 'chimerax_boltz22', wait = None):
    if directory is None:
        from os.path import expanduser
        directory = expanduser('~/boltz22')

    # Check that directory either does not exist or is empty.
    from os.path import exists, isdir
    if exists(directory):
        from os import listdir
        if not isdir(directory) or len(listdir(directory)):
            from chimerax.core.errors import UserError
            raise UserError(f'You must install Boltz into a new or empty directory.  The directory {directory} already exists and is not empty.')

    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'x86_64':
        from chimerax.core.errors import UserError
        raise UserError('Boltz requires newer Torch versions that are not available on Intel Macs.')

    if wait is None:
        wait = False if session.ui.is_gui else True

    ib = InstallBoltz(session, directory, download_model_weights_and_ccd,
                      branch = branch, wait = wait)
    return ib
            
# ------------------------------------------------------------------------------
#
class InstallBoltz:

    def __init__(self, session, directory, download_model_weights_and_ccd = True,
                 branch = 'chimerax', wait = False):

        self._session = session
        self._directory = directory
        self._download_model_weights_and_ccd = download_model_weights_and_ccd
        self._branch = branch		# Git branch to install
        self._wait = wait
        self.finished_callback = None
        self.success = None

        if self._create_boltz_virtual_environment(directory):
            self._install_boltz()
        else:
            success = False

    # ------------------------------------------------------------------------------
    #
    def _create_boltz_virtual_environment(self, directory):
        # Create Python virtual environment using ChimeraX Python as base for installing Boltz.
        from chimerax.core.python_utils import chimerax_python_executable
        python_exe = chimerax_python_executable()
        command = [python_exe, '-m', 'venv', '--system-site-packages', directory]
        from subprocess import run
        p = run(command, capture_output = True, creationflags = _no_subprocess_window())

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
    def _venv_python_executable(self):
        return find_executable(self._directory, 'python')

    # ------------------------------------------------------------------------------
    #
    def _install_boltz(self):
        if self._need_cuda_torch_on_windows():
            # The standard PyPi torch is cpu only, so get cuda-enabled torch from pytorch.org.
            self._pip_install_cuda_torch()
        else:
            self._pip_install_boltz()

    # ------------------------------------------------------------------------------
    #
    def _need_cuda_torch_on_windows(self):
        return have_nvidia_driver()

    # ------------------------------------------------------------------------------
    #
    def _pip_install_cuda_torch(self):
        # Run pip install of torch in virtual environment
        logger = self._session.logger
        logger.info('Now installing machine learning package torch.')
        # TODO: We should try to match the system cuda version.
        command = [self._venv_python_executable(), '-m', 'pip', 'install', 'torch==2.7.1',
                   '--index-url', 'https://download.pytorch.org/whl/cu126']
        logger.info(' '.join(command))

        # Echo subprocess output to the ChimeraX Log.
        log_subprocess_output(self._session, command, self._finished_install_cuda_torch, wait = self._wait)
    
    # ------------------------------------------------------------------------------
    #
    def _finished_install_cuda_torch(self, success):
        # Report success of failure of pip install of torch
        logger = self._session.logger
        if success:
            logger.info('Successfully installed torch.')
        else:
            logger.error('Torch installation failed.  See ChimeraX Log for details.')
        self._finished('pip install torch', success = success)

    # ------------------------------------------------------------------------------
    #
    def _pip_install_boltz(self):
        # Run pip install of Boltz in virtual environment
        logger = self._session.logger
        logger.info('Now installing Boltz and required packages from PyPi.  This may take tens of of minutes'
                    ' since Boltz uses many other packages totaling about 1 Gbyte of disk'
                    ' space including torch, scipy, rdkit, llvmlite, sympy, pandas, numpy, wandb, numba...')

#        boltz_ver = 'boltz==0.4.1'
#        boltz_ver = 'git+https://github.com/jwohlwend/boltz@a9b3abc2c1f90f26b373dd1bcb7afb5a3cb40293'  # Install from Github source
#        boltz_ver = f'git+https://github.com/RBVI/boltz@{self._branch}'  # Install from RBVI fork of Boltz
        boltz_ver = f'https://github.com/RBVI/boltz/archive/{self._branch}.zip'  # Install from RBVI fork of Boltz using zip so git not needed.
        command = [self._venv_python_executable(), '-m', 'pip', 'install', boltz_ver]
        logger.info(' '.join(command))

        # Echo subprocess output to the ChimeraX Log.
        log_subprocess_output(self._session, command, self._finished_pip_install_boltz, wait = self._wait)
    
    # ------------------------------------------------------------------------------
    #
    def _finished_pip_install_boltz(self, success):
        # Report success of failure of pip install of Boltz
        logger = self._session.logger
        if success:
            # Remember the Boltz install directory
            from .settings import _boltz_settings
            settings = _boltz_settings(self._session)
            settings.boltz22_install_location = self._directory
            settings.save()

            # Make the Boltz GUI show the install location.
            from .boltz_gui import boltz_panel
            p = boltz_panel(self._session)
            if p:
                p._install_directory.value = self._directory
        else:
            logger.error('Boltz installation failed.  See ChimeraX Log for details.')

        self._finished('pip install boltz', success = success)

    # ------------------------------------------------------------------------------
    #
    def _download_model_weights_and_ccd_database(self):

        logger = self._session.logger
        logger.info('Downloading Boltz model parameters (4 GB) and chemical component database (1.8 GB) to ~/.boltz')

        from os.path import join, dirname
        download_path = join(dirname(__file__), 'download_weights_and_ccd.py')
        command = [self._venv_python_executable(), download_path]
        from sys import platform
        if platform == 'darwin':
            # On Mac the huggingface.co URLs get SSL certificate errors unless we setup certifi root certificates.
            import certifi
            env = {"SSL_CERT_FILE": certifi.where()}
        else:
            env = None
        logger.info(' '.join(command))

        # Echo subprocess output to the ChimeraX Log.
        log_subprocess_output(self._session, command, self._finished_download_weights_and_ccd,
                              wait = self._wait)

    # ------------------------------------------------------------------------------
    #
    def _finished_download_weights_and_ccd(self, success):
        # Report success of failure downloading boltz model parameters and ccd
        logger = self._session.logger
        if success:
            logger.info('Boltz model parameters and CCD database are installed in ~/.boltz')
        else:
            logger.error('Downloading Boltz model parameters and CCD database to ~/.boltz or creating CCD atom counts file failed.')

        self._finished('download weights and ccd', success)

    # ------------------------------------------------------------------------------
    #
    def _finished(self, task_name, success = True):
        if success:
            if task_name == 'pip install torch':
                self._pip_install_boltz()
                return
            elif task_name == 'pip install boltz':
                if self._download_model_weights_and_ccd:
                    self._download_model_weights_and_ccd_database()
                    return
            self._session.logger.info('Successfully installed Boltz program and neural net weights.')

        self.success = success
        if self.finished_callback:
            self.finished_callback(success)

# ------------------------------------------------------------------------------
#
class log_subprocess_output:
    def __init__(self, session, command, finished_callback, wait = False):
        self._session = session

        from subprocess import Popen, PIPE, STDOUT
        popen = Popen(command, stdout = PIPE, stderr = STDOUT, creationflags = _no_subprocess_window())
        self._popen = popen

        self._finished_callback = finished_callback
        from queue import Queue
        self._queue = Queue()
        from threading import Thread
        # Set daemon true so that ChimeraX exit is not blocked by the thread still running.
        self._thread = t = Thread(target = self._queue_output_in_thread, daemon = True)
        t.start()
        if wait:
            while t.is_alive():
                self._log_queued_lines()
        else:
            session.triggers.add_handler('new frame', self._log_queued_lines)

    def _queue_output_in_thread(self):
        while True:
            line = self._popen.stdout.readline() # blocking read
            if not line:
                break
            self._queue.put(line)

    def _log_queued_lines(self, *trigger_args):
        while not self._queue.empty():
            line = self._queue.get()
            import locale
            stdout_encoding = locale.getpreferredencoding()
            self._session.logger.info(line.decode(stdout_encoding, errors = 'ignore'))
        if not self._thread.is_alive():
            self._finished()
            return 'delete handler'

    def _finished(self):
        self._popen.wait()  # Set returncode
        success = (self._popen.returncode == 0)
        self._finished_callback(success)

# ------------------------------------------------------------------------------
#
def have_nvidia_driver():
    from sys import platform
    if platform == 'win32':
        nvidia_smi_path = 'C:\\Windows\\System32\\nvidia-smi.exe'
    elif platform == 'linux':
        nvidia_smi_path = '/usr/bin/nvidia-smi'
    else:
        return False
    from os.path import exists
    return exists(nvidia_smi_path)

# ------------------------------------------------------------------------------
#
def _no_subprocess_window():
    '''The Python subprocess module only has the CREATE_NO_WINDOW flag on Windows.'''
    from sys import platform
    if platform == 'win32':
        from subprocess import CREATE_NO_WINDOW
        flags = CREATE_NO_WINDOW
    else:
        flags = 0
    return flags
            
# ------------------------------------------------------------------------------
#
def find_executable(venv_directory, exe_name):
    from os.path import join
    from sys import platform
    if platform == 'win32':
        exe = join(venv_directory, 'Scripts', exe_name + '.exe')
    else:
        exe = join(venv_directory, 'bin', exe_name)
    return exe
            
# ------------------------------------------------------------------------------
#
def register_boltz_install_command(logger):
    from chimerax.core.commands import CmdDesc, register, SaveFolderNameArg, BoolArg, StringArg
    desc = CmdDesc(
        optional = [('directory', SaveFolderNameArg)],
        keyword = [('download_model_weights_and_ccd', BoolArg),
                   ('branch', StringArg)],
        synopsis = 'Install Boltz from PyPi in a virtual environment',
        url = 'help:boltz_help.html'
    )
    register('boltz install', desc, boltz_install, logger=logger)
