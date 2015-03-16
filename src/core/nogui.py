# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
nogui: Text UI
==============

Text-based user interface.  API-compatible with :py:module:`ui` package.
"""
from .utils import flattened
from .tasks import Task
from .logger import PlainTextLog


class NoGuiLog(PlainTextLog):

    def log(self, level, msg):
        print("%s: %s" % (level, msg))
        return True

    def status(self, msg, color, secondary):
        if secondary:
            return False
        print("status: %s" % msg)
        return True


class UI:

    def __init__(self, session):
        session.logger.add_log(NoGuiLog())
        import weakref
        self._session = weakref.ref(session)
        self._queue = None

    def splash_info(self, message, splash_step, num_splash_steps):
        import sys
        print("%.2f%% done: %s" % (splash_step / num_splash_steps * 100,
                                   message), file=sys.stderr)

    def build(self):
        pass  # nothing to build

    def quit(self):
        import os
        import sys
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
                break

    def run_command(self, text):
        # Run command from input queue, runs in UI thread
        # Separate from "execute" to handle input synchronization
        self.execute(text)
        self._sem.release()

    def execute(self, text):
        # Command execution, runs in UI thread
        from . import cli
        try:
            self._cmd.parse_text(text, final=True)
            results = self._cmd.execute()
            for result in flattened(results):
                if result is not None:
                    print(result)
        except cli.UserError as err:
            print(self._cmd.current_text)
            rest = self._cmd.current_text[self._cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = self._cmd.amount_parsed + spaces
            print("%s^" % ('.' * error_at))
            print(err)

    #
    # Required State methods, do nothing
    #
    def reset_state(self):
        pass

    def take_snapshot(self, session, flags):
        return [1, None]

    def restore_snapshot(self, phase, session, version, data):
        pass
