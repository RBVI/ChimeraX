# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
nogui: Text UI
==============

Text-based user interface.  API-compatible with :py:module:`ui` package.
"""
from .tasks import Task
from .logger import PlainTextLog
_color_output = False


class NoGuiLog(PlainTextLog):

    def log(self, level, msg):
        if not _color_output:
            print("%s: %s" % (level, msg))
        else:
            from colorama import Fore
            if level == self.LEVEL_INFO:
                print(Fore.GREEN + msg + Fore.RESET, end='')
            elif level == self.LEVEL_ERROR:
                print(Fore.RED + msg + Fore.RESET, end='')
            elif level == self.LEVEL_WARNING:
                print(Fore.YELLOW + msg + Fore.RESET, end='')
        return True

    def status(self, msg, color, secondary):
        if secondary:
            return False
        if msg:
            if not _color_output:
                print("status: %s" % msg)
            else:
                from colorama import Fore
                print(Fore.MAGENTA + msg + Fore.RESET)
        return True


class UI:

    def __init__(self, session):
        global _color_output
        self.is_gui = False
        session.logger.add_log(NoGuiLog())
        import weakref
        self._session = weakref.ref(session)
        self._queue = None
        import sys
        if sys.stdout.isatty():
            try:
                import colorama
                colorama.init()
                _color_output = True
                print(colorama.Fore.WHITE + colorama.Back.BLACK, end='')
            except ImportError:
                pass

        if not self.initialize_offscreen_rendering(session.main_view):
            session.logger.info('Offscreen rendering not available')

    def initialize_offscreen_rendering(self, view):
        from . import graphics
        try:
            c = graphics.OffScreenRenderingContext()
        except:
            c = None             # OSMesa library was not found
        if c:
            view.initialize_context(c)
        return not c is None

    def splash_info(self, message, splash_step, num_splash_steps):
        import sys
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
    SESSION_SKIP = True

    def __init__(self, session):
        # Initializer, runs in UI thread
        super().__init__(session)
        from . import cli
        self._cmd = cli.Command(session)
        from threading import Semaphore
        self._sem = Semaphore()

    def run(self):
        # Actual event loop, runs in our own thread
        # Schedules calls to self.execute in UI thread
        prompt = 'cmd> '
        if _color_output:
            from colorama import Fore, Style
            prompt = Fore.GREEN + Style.BRIGHT + prompt + Fore.RESET
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
        from . import cli
        text = text.strip()
        if not text:
            return
        try:
            self._cmd.parse_text(text, final=True)
            self._cmd.execute()
        except cli.UserError as err:
            logger = self.session.logger
            logger.info(self._cmd.current_text)
            rest = self._cmd.current_text[self._cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = self._cmd.amount_parsed + spaces
            if error_at < len(self._cmd.current_text):
                logger.error("%s^" % ('.' * error_at))
            logger.error(str(err))
        except Exception:
            logger = self.session.logger
            logger.error("\nUnexpected exception, save your work and exit:\n")
            import traceback
            logger.error(traceback.format_exc())

    #
    # Required State methods, do nothing
    #
    def reset_state(self):
        pass

    def take_snapshot(self, phase, session, flags):
        return

    def restore_snapshot(self, phase, session, version, data):
        pass
