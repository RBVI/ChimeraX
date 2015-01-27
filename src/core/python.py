# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
py: Python file support
=======================

Read and execute Python scripts
"""


def open_py(session, filename, *args, **kw):
    name = kw['name'] if 'name' in kw else None
    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
        if name is None:
            name = filename.name
    else:
        input = _builtin_open(filename, 'rb')
        if name is None:
            name = filename

    try:
        data = input.read()
        code = compile(data, name, 'exec')
        global_dict = {
            '__name__': '%s_sandbox' % sesssion.app_dirs.appname,
            '%s_session' % session.app_dirs.appname: session
        }
        exec(code, global_dict)
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
