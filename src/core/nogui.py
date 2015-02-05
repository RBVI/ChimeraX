# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
nogui: Text UI
==============

Text-based user interface.  API-compatible with :py:module:`ui` package.
"""
from .utils import flattened
from .tasks import Task


class UI:

    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)
        from . import cli
        self._cmd = cli.Command(session)
        from threading import Semaphore
        self._input_sem = Semaphore()
        from queue import Queue
        self._queue = Queue()

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
        session = self._session()  # resolve back reference
        input = Input(session, self._input_sem)
        input.start()
        from .tasks import FINISHED, TERMINATED
        while input.state not in [FINISHED, TERMINATED]:
            func, args, kw = self._queue.get()
            func(*args, **kw)

    def thread_safe(self, func, *args, **kw):
        self._queue.put((func, args, kw))

    def execute(self, text):
        from . import cli
        try:
            self._cmd.parse_text(text, final=True)
            results = self._cmd.execute()
            for result in flattened(results):
                if result is not None:
                    print(result)
        except cli.UserError as err:
            print(cmd.current_text)
            rest = cmd.current_text[cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = cmd.amount_parsed + spaces
            print("%s^" % ('.' * error_at))
            print(err)
        self._input_sem.release()


class Input(Task):

    def __init__(self, session, sem):
        super().__init__(session)
        self._sem = sem

    def run(self):
        prompt = 'cmd> '
        ui = self.session.ui
        while True:
            try:
                self._sem.acquire()
                text = input(prompt)
                ui.thread_safe(ui.execute, text)
            except EOFError:
                ui.thread_safe(ui.quit)

    def reset_state(self):
        pass

    def take_snapshot(self, session, flags):
        return [1, None]

    def restore_snapshot(self, phase, session, version, data):
        pass
