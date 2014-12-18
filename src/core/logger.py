# vim: set expandtab ts=4 sw=4:


class Logger:

    def __init__(self):
        self.logs = []

    def error(self, msg, add_newline=True, image=None, is_html=False):
        end = "\n" if add_newline else ""
        import sys
        print("error:", msg, end=end, file=sys.stderr)

    def warning(self, msg, add_newline=True, image=None, is_html=False):
        end = "\n" if add_newline else ""
        import sys
        print("warning:", msg, end=end, file=sys.stderr)

    def info(self, msg, add_newline=True, image=None, is_html=False):
        end = "\n" if add_newline else ""
        print(msg, end=end)

    def status(self, msg, **kw):
        print('status:', msg)
