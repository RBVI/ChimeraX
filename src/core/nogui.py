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

"""
nogui: Text UI
==============

Text-based user interface.  API-compatible with :py:module:`ui` package.
"""
from .tasks import Task
from .logger import PlainTextLog
_color_output = None

_log_level = {
    PlainTextLog.LEVEL_INFO: 'info',
    PlainTextLog.LEVEL_WARNING: 'warning',
    PlainTextLog.LEVEL_ERROR: 'error',
    PlainTextLog.LEVEL_BUG: 'bug',
}

_colors = {
    "info": "",
    "warning": "",
    "error": "",
    "bug": "",
    "status": "",
    "normal": "",
    "background": "",
    "prompt": "",
    "endprompt": "",
}


class NoGuiLog(PlainTextLog):

    def log(self, level, msg):
        level_name = _log_level[level]
        import sys
        encoding = sys.stdout.encoding.lower()
        if encoding != 'utf-8' and isinstance(msg, str):
            msg = msg.encode(encoding, 'replace').decode(encoding)

        if _color_output:
            print("%s%s%s" % (
                _colors[level_name], msg, _colors["normal"]), end='', flush=True)
        else:
            print("%s:\n%s" % (level_name.upper(), msg), end='', flush=True)
        return True

    def status(self, msg, color, secondary):
        if secondary:
            return False
        if msg:
            if _color_output:
                print("%s%s%s" % (_colors["status"], msg, _colors["normal"]), flush=True)
            else:
                print("STATUS:\n%s" % msg, flush=True)
        return True


class UI:

    def __init__(self, session):
        self.is_gui = False
        self.has_graphics = False
        session.logger.add_log(NoGuiLog())

        import weakref
        self._session = weakref.ref(session)
        self._queue = None

    def initialize_color_output(self, color_output):
        global _color_output
        _color_output = color_output
        import sys
        if _color_output or (_color_output is None and sys.stdout.isatty()):
            try:
                import colorama
                colorama.init()
                _color_output = True
                _colors["info"] = colorama.Fore.GREEN
                _colors["warning"] = colorama.Fore.YELLOW
                _colors["error"] = colorama.Fore.RED
                _colors["bug"] = colorama.Fore.RED + colorama.Style.BRIGHT
                _colors["status"] = colorama.Fore.MAGENTA
                _colors["normal"] = colorama.Fore.WHITE
                _colors["background"] = colorama.Back.BLACK
                _colors["prompt"] = colorama.Fore.BLUE + colorama.Style.BRIGHT
                _colors["endprompt"] = colorama.Style.NORMAL + _colors["normal"]
                # Hack around colorama not checking for closed streams at exit
                import atexit
                import colorama.initialise
                atexit.unregister(colorama.initialise.reset_all)

                def reset():
                    try:
                        colorama.initialise.reset_all()
                    except ValueError:
                        # raised if stream is closed
                        pass
                atexit.register(reset)
            except ImportError:
                pass

    def initialize_offscreen_rendering(self):
        session = self._session()
        from chimerax import graphics
        try:
            c = graphics.OffScreenRenderingContext()
        except Exception as e:
            # OSMesa library was not found, or old version
            if not session.silent:
                session.logger.info('Offscreen rendering is not available.')
                session.logger.info(str(e))
            self.has_graphics = False
            return
        session.main_view.initialize_rendering(c)
        self.has_graphics = True

        # Create an offscreen QApplication so labels will work
        try:
            from Qt.QtWidgets import QApplication
        except ModuleNotFoundError:
            return	# ChimeraX being used without Qt
        import chimerax
        app_name = chimerax.app_dirs.appname if hasattr(chimerax, 'app_dirs') else 'ChimeraX'
        self._app = QApplication([app_name, '-platform', 'offscreen'])

    def splash_info(self, message, splash_step=None, num_splash_steps=None):
        import sys
        if splash_step is None:
            print(message, file=sys.stderr)
        else:
            print("%.2f%% done: %s" % (splash_step / num_splash_steps * 100,
                                       message), file=sys.stderr)

    def build(self):
        pass  # nothing to build

    def quit(self):
        import os
        import sys
        session = self._session()  # resolve back reference
        session.logger.status("Exiting ...", blank_after=0)
        session.logger.clear()   # Clear logging timers
        sys.exit(os.EX_OK)

    def event_loop(self):
        from queue import Queue
        self._queue = Queue()
        session = self._session()  # resolve back reference
        input = _Input(session)
        input.start()
        from .tasks import FINISHED, TERMINATED
        while input.state not in [FINISHED, TERMINATED]:
            func, args, kw = self._queue.get()
            try:
                func(*args, **kw)
            finally:
                self._queue.task_done()

    def thread_safe(self, func, *args, **kw):
        if self._queue:
            self._queue.put((func, args, kw))
        else:
            func(*args, **kw)


class _Input(Task):

    SESSION_ENDURING = True

    def __init__(self, session):
        # Initializer, runs in UI thread
        super().__init__(session)
        from .commands import Command
        self._cmd = Command(session)
        from threading import Semaphore
        self._sem = Semaphore()

    def run(self):
        # Actual event loop, runs in our own thread
        # Schedules calls to self.execute in UI thread
        prompt = 'cmd> '
        if _color_output:
            prompt = _colors["prompt"] + prompt + _colors["endprompt"]
        ui = self.session.ui
        while True:
            try:
                self._sem.acquire()
                text = input(prompt)
                ui.thread_safe(self.run_command, text)
            except EOFError:
                # Need to get UI thread to do something
                # in order to detect termination of input thread
                ui.thread_safe(print, "EOF")
                self.session.logger.clear()
                break

    def run_command(self, text):
        # Run command from input queue, runs in UI thread
        # Separate from "execute" to handle input synchronization
        self.execute(text)
        self._sem.release()

    def execute(self, text):
        # Command execution, runs in UI thread
        from . import errors
        text = text.strip()
        if not text:
            return
        logger = self.session.logger
        try:
            self._cmd.run(text)
        except errors.NotABug as err:
            logger.error(str(err))
        except Exception:
            logger.error("\nUnexpected exception, save your work and exit:\n")
            import traceback
            logger.bug(traceback.format_exc())

    #
    # Required State methods, do nothing
    #
    def take_snapshot(self, session, flags):
        return

    @staticmethod
    def restore_snapshot(session, data):
        pass
