# SPDX-License-Identifier: MIT
# Copyright 2026 Samuel Lobo

from os.path import exists, expanduser, isdir, join

from chimerax.core.errors import UserError

from .settings import _fasthydromap_settings


DEFAULT_PACKAGE_SPEC = "fasthydromap>=0.1.4,<0.2"
DEFAULT_TORCH_VARIANT = "cpu"


def fasthydromap_install(
    session,
    directory=None,
    *,
    wait=None,
    torch_variant=DEFAULT_TORCH_VARIANT,
    package_spec=DEFAULT_PACKAGE_SPEC,
):
    if directory is None:
        from chimerax import app_dirs

        directory = join(app_dirs.user_data_dir, "fasthydromap")
    else:
        directory = expanduser(directory)

    if wait is None:
        wait = False if session.ui.is_gui else True

    return InstallFastHydroMap(
        session,
        directory,
        package_spec=package_spec,
        torch_variant=torch_variant,
        wait=wait,
    )


class InstallFastHydroMap:
    def __init__(self, session, directory, *, package_spec, torch_variant, wait):
        self._session = session
        self._directory = directory
        self._package_spec = package_spec
        self._torch_variant = torch_variant
        self._wait = wait
        self.finished_callback = None
        self.success = None

        self._check_install_directory()
        if self._create_virtual_environment():
            self._upgrade_packaging_tools()

    def _check_install_directory(self):
        if exists(self._directory):
            from os import listdir

            if not isdir(self._directory) or listdir(self._directory):
                raise UserError(
                    "You must install FastHydroMap into a new or empty directory. "
                    f"The directory {self._directory} already exists and is not empty."
                )

    def _create_virtual_environment(self):
        from chimerax.core.python_utils import chimerax_python_executable
        from subprocess import run

        python_exe = chimerax_python_executable()
        if _is_macos_app_translocation(python_exe):
            self._session.logger.error(
                "Cannot install FastHydroMap while ChimeraX is running from a temporary "
                "macOS App Translocation path. Install ChimeraX in Applications with Finder, "
                "quit ChimeraX, relaunch the installed app, and run 'fasthydromap install' again."
            )
            self._finished("create environment", success=False)
            return False
        command = [python_exe, "-m", "venv", self._directory]
        p = run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                creationflags=_no_subprocess_window())

        logger = self._session.logger
        if p.returncode == 0:
            logger.info(f"Successfully created FastHydroMap Python virtual environment {self._directory}.")
            return True

        logger.error(
            "Creating FastHydroMap Python virtual environment failed."
            f"\nCommand: {' '.join(command)}"
            f"\nstdout: {p.stdout}"
            f"\nstderr: {p.stderr}"
        )
        self._finished("create environment", success=False)
        return False

    def _venv_python_executable(self):
        return find_executable(self._directory, "python")

    def _fasthydromap_executable(self):
        return find_executable(self._directory, "fasthydromap")

    def _upgrade_packaging_tools(self):
        logger = self._session.logger
        logger.info("Upgrading pip/setuptools/wheel in the FastHydroMap environment.")
        command = [
            self._venv_python_executable(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools<81",
            "wheel",
        ]
        log_subprocess_output(self._session, command, self._finished_upgrade_packaging_tools, wait=self._wait)

    def _finished_upgrade_packaging_tools(self, success):
        if success:
            self._install_fasthydromap_package()
        else:
            self._session.logger.error("Failed to upgrade packaging tools for FastHydroMap.")
            self._finished("upgrade packaging tools", success=False)

    def _install_fasthydromap_package(self):
        logger = self._session.logger
        logger.info(f"Installing FastHydroMap package {self._package_spec} from PyPI.")
        command = [
            self._venv_python_executable(),
            "-m",
            "pip",
            "install",
            self._package_spec,
        ]
        log_subprocess_output(
            self._session,
            command,
            self._finished_install_fasthydromap_package,
            wait=self._wait,
        )

    def _finished_install_fasthydromap_package(self, success):
        if success:
            self._install_torch()
        else:
            self._session.logger.error("Failed to install FastHydroMap from PyPI.")
            self._finished("install package", success=False)

    def _install_torch(self):
        logger = self._session.logger
        logger.info(
            f"Installing Torch into the managed FastHydroMap environment using variant "
            f"{self._torch_variant}."
        )
        command = [
            self._fasthydromap_executable(),
            "install-torch",
            "--variant",
            self._torch_variant,
        ]
        log_subprocess_output(
            self._session,
            command,
            self._finished_install_torch,
            wait=self._wait,
        )

    def _finished_install_torch(self, success):
        if success:
            self._finalize_install()
        else:
            self._session.logger.error("Failed to install Torch for FastHydroMap.")
            self._finished("install torch", success=False)

    def _finalize_install(self):
        self._finish_install(success=True)

    def _finish_install(self, success):
        logger = self._session.logger
        if success:
            settings = _fasthydromap_settings(self._session)
            settings.fasthydromap_install_location = self._directory
            settings.save()

            exe = self._fasthydromap_executable()
            if exists(exe):
                logger.info(f"FastHydroMap executable installed at {exe}")
                self._finished("install package", success=True)
                return

            logger.error(
                "FastHydroMap package installed, but the fasthydromap executable was not found "
                f"in {self._directory}."
            )
        else:
            logger.error("FastHydroMap installation failed.  See ChimeraX Log for details.")

        self._finished("install package", success=False)

    def _finished(self, task_name, success=True):
        if success:
            _log_ready_message(self._session)
        self.success = success
        if self.finished_callback:
            self.finished_callback(success)


class log_subprocess_output:
    def __init__(self, session, command, finished_callback, wait=False):
        self._session = session

        from subprocess import PIPE, STDOUT, Popen

        popen = Popen(command, stdout=PIPE, stderr=STDOUT, creationflags=_no_subprocess_window())
        self._popen = popen

        self._finished_callback = finished_callback
        from queue import Queue
        from threading import Thread

        self._queue = Queue()
        self._thread = t = Thread(target=self._queue_output_in_thread, daemon=True)
        t.start()
        if wait:
            while t.is_alive():
                self._log_queued_lines()
            self._finished()
        else:
            session.triggers.add_handler("new frame", self._log_queued_lines_while_alive)

    def _queue_output_in_thread(self):
        while True:
            line = self._popen.stdout.readline()
            if not line:
                break
            self._queue.put(line)

    def _log_queued_lines(self):
        while not self._queue.empty():
            line = self._queue.get()
            self._session.logger.info(line.decode("utf-8", errors="replace").rstrip())

    def _log_queued_lines_while_alive(self, *trigger_args):
        self._log_queued_lines()
        if not self._thread.is_alive():
            self._finished()
            return "delete handler"

    def _finished(self):
        self._popen.wait()
        success = self._popen.returncode == 0
        self._finished_callback(success)

def managed_fasthydromap_executable(session, install_location=None):
    location = install_location
    if location is None:
        settings = _fasthydromap_settings(session)
        location = settings.fasthydromap_install_location
    if not location:
        return None
    exe = find_executable(location, "fasthydromap")
    return exe if exists(exe) else None


def find_executable(venv_directory, exe_name):
    from sys import platform

    if platform == "win32":
        return join(venv_directory, "Scripts", exe_name + ".exe")
    return join(venv_directory, "bin", exe_name)


def _no_subprocess_window():
    from sys import platform

    if platform == "win32":
        from subprocess import CREATE_NO_WINDOW

        return CREATE_NO_WINDOW
    return 0


def _is_macos_app_translocation(executable):
    from sys import platform

    return platform == "darwin" and "/AppTranslocation/" in executable


def _log_ready_message(session):
    session.logger.info(
        "<b>FastHydroMap is ready.</b><br>"
        "Try: "
        '<a href="cxcmd:open 1a1u">open 1a1u</a>, then '
        '<a href="cxcmd:fasthydromap #1">fasthydromap #1</a>.<br>'
        "Explore water structure with "
        '<a href="cxcmd:fasthydromap #1 quantity pc1">'
        "fasthydromap #1 quantity pc1</a>, or open "
        '<a href="cxcmd:help fasthydromap">help fasthydromap</a> '
        "for examples and interpretation.",
        is_html=True,
    )


def register_fasthydromap_install_command(logger):
    from chimerax.core.commands import BoolArg, CmdDesc, SaveFolderNameArg, StringArg, register

    desc = CmdDesc(
        optional=[("directory", SaveFolderNameArg)],
        keyword=[
            ("wait", BoolArg),
            ("torch_variant", StringArg),
            ("package_spec", StringArg),
        ],
        synopsis="Install FastHydroMap from PyPI in a managed ChimeraX virtual environment",
    )
    register("fasthydromap install", desc, fasthydromap_install, logger=logger)
