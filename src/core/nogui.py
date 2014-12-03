# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
nogui: Text UI
==============

Text-based user interface.  API-compatible with :py:module:`ui` package.
"""


# from Mike C. Fletcher's BasicTypes library
# via http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
def _flattened(l, ltypes=(list, tuple, set)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


class UI:

    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)

    def splash_info(self, message, splash_step, num_splash_steps):
        import sys
        print("%.2f%% done: %s" % (splash_step / num_splash_steps * 100,
                                   message), file=sys.stderr)

    def build(self):
        # nothing to build

        # register a few commands for debugging
        from . import cli

        @cli.register('exit')
        def exit(session):
            raise SystemExit(0)

        @cli.register('echo', cli.CmdDesc(optional=[('text', cli.RestOfLine)]))
        def echo(session, text=''):
            return text

    def event_loop(self):
        session = self._session()  # resolve back reference
        prompt = 'cmd> '
        from . import cli
        cmd = cli.Command(session)
        while True:
            try:
                text = input(prompt)
                cmd.parse_text(text, final=True)
                results = cmd.execute()
                for result in _flattened(results):
                    if result is not None:
                        print(result)
            except EOFError:
                raise SystemExit(0)
            except cli.UserError as err:
                print(cmd.current_text)
                rest = cmd.current_text[cmd.amount_parsed:]
                spaces = len(rest) - len(rest.lstrip())
                error_at = cmd.amount_parsed + spaces
                print("%s^" % ('.' * error_at))
                print(err)
