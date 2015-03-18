# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
py: Python file support
=======================

Read and execute Python scripts
"""

_builtin_open = open


def open_py(session, filename, name, *args, **kw):
    """Execute Python file in Chimera context"""
    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    try:
        data = input.read()
        code = compile(data, name, 'exec')
        global_dict = {
            '__name__': '%s_sandbox' % session.app_dirs.appname,
            '%s_session' % session.app_dirs.appname: session
        }
        exec(code, global_dict)
    finally:
        if input != filename:
            input.close()
    return [], "executed %s" % name

def open_ch(session, filename, name, *args, **kw):
    """Execute Python file in Chimera commands"""
    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    from .cli import Command
    try:
        for line in input.readlines():
            text = line.strip().decode('utf-8', errors='replace')
            cmd = Command(session, text, final=True)
            cmd.execute()
    finally:
        if input != filename:
            input.close()
    return [], "executed %s" % name


def register():
    from . import io
    io.register_format(
        "Python", io.SCRIPT, (".py", ".pyc", ".pyo"), ("py",),
        mime=('text/x-python', 'application/x-python-code'),
        reference="http://www.python.org/",
        open_func=open_py)
    io.register_format(
        "Chimera", io.SCRIPT, (".c2cmd",), ("cmd",),
        mime=('text/x-chimera2', 'application/x-chimera2-code'),
        reference="http://www.cgl.ucsf.edu/chimera/",
        open_func=open_ch)
