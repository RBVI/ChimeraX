# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
scripting: support reading and executing scripts
================================================

Both Python and Chimera2 command scripts are supported.

Python scripts are executed inside a sandbox module that has
the Chimera2 session available in it.
For example, to use the timeit module in a Python script::

    import timeit
    from chimera.core.comands import sym

    m = session.models.list()[0]
    t = timeit.timeit(
        "sym.pdb_assemblies(m)",
        "from %s import sym, m" % __name__,
        number=1000
    )
    print('total time:', t)
"""

_builtin_open = open
_sandbox_count = 0


def open_py(session, filename, name, *args, **kw):
    """Execute Python script in a Chimera2 context

    This function is invoked via Chimera2's :py:mod:`~chimera.core.io`
    :py:func:`~chimera.core.io.open_data` API for files whose names end
    with **.py**, **.pyc**, or **.pyo**.  Each script is opened in an uniquely
    named importable sandbox (see timeit example above).  And the current
    Chimera2 session is available as a global variable named **session**.
    
    Parameters
    ----------
    session : a Chimera2 :py:class:`~chimera.core.session.Session`
    filename : path to file to open
    name : how to identify the file
    """
    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    try:
        data = input.read()
        code = compile(data, name, 'exec')
        import sys
        import types
        global _sandbox_count
        _sandbox_count += 1
        sandbox = types.ModuleType(
            '%s_sandbox_%d' % (session.app_dirs.appname, _sandbox_count),
            '%s script sandbox' % session.app_dirs.appname)
        setattr(sandbox, 'session', session)
        try:
            sys.modules[sandbox.__name__] = sandbox
            exec(code, sandbox.__dict__)
        finally:
            del sys.modules[sandbox.__name__]
    finally:
        if input != filename:
            input.close()
    return [], "executed %s" % name


def open_ch(session, filename, name, *args, **kw):
    """Execute utf-8 file as Chimera2 commands

    This function is invoked via Chimera2's :py:mod:`~chimera.core.io`
    :py:func:`~chimera.core.io.open_data` API for files whose names end
    with **.c2cmd**.
    
    Parameters
    ----------
    session : a Chimera2 :py:class:`~chimera.core.session.Session`
    filename : path to file to open
    name : how to identify the file
    
    """
    if hasattr(filename, 'read'):
        # it's really a fetched stream
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    from .commands import run
    try:
        for line in input.readlines():
            text = line.strip().decode('utf-8', errors='replace')
            cmd = run(session, text)
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
