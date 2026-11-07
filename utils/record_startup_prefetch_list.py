# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Record the startup prefetch list to ship with a built ChimeraX.

Run with the built application's launcher, which names the list to record,
after the GUI is up:

    ChimeraX.app/bin/ChimeraX --script utils/record_startup_prefetch_list.py

It records this start's files as chimerax.core.startup_prefetch does after
every GUI start, converts the list to names relative to the application
directory (make_startup_prefetch_list.py), writes it to the application's
share/startup-prefetch, which the Linux and Windows launchers read until a
user's own list exists, and exits.
"""
import os
import sys


def _app_dir():
    executable = os.path.realpath(sys.executable)
    if sys.platform == "win32":
        return os.path.dirname(executable)                    # ChimeraX.app\bin
    return os.path.dirname(os.path.dirname(executable))       # ChimeraX.app


def _record():
    from chimerax.core import startup_prefetch
    path = os.environ.get(startup_prefetch.ENV_VAR)
    if not path:
        session.logger.error("Start ChimeraX with its launcher to record a prefetch list.")
        os._exit(1)
    thread = startup_prefetch.record_startup_files()
    if thread is not None:
        thread.join(60)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_startup_prefetch_list",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_startup_prefetch_list.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    app_dir = _app_dir()
    output = os.path.join(app_dir, "share", "startup-prefetch")
    with open(path, encoding="utf-8") as f:
        text = tool.relative_list(f.read(), app_dir)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{output}: {text.count(chr(10))} entries", file=sys.__stdout__, flush=True)
    os._exit(0)


def _when_ready():
    # The same moment startup records its own list: after the GUI is up.
    from Qt.QtCore import QTimer
    QTimer.singleShot(3000, _record)


_when_ready()
