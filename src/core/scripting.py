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
scripting: support reading and executing scripts
================================================

Both Python and ChimeraX command scripts are supported.

Python scripts are executed inside a sandbox module that has
the ChimeraX session available in it.
For example, to use the timeit module in a Python script::

    import timeit
    from chimerax.core.comands import sym

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


def _exec_python(session, code, argv=None):
    # actual routine that sandboxes executing Python code
    import sys
    import types
    from chimerax import app_dirs
    global _sandbox_count
    _sandbox_count += 1
    sandbox = types.ModuleType(
        '%s_sandbox_%d' % (app_dirs.appname, _sandbox_count),
        '%s script sandbox' % app_dirs.appname)
    if argv is None:
        restore_argv = False
    else:
        restore_argv = True
        orig_argv = sys.argv
        sys.argv = argv
    setattr(sandbox, 'session', session)
    if hasattr(code, 'co_filename'):
        setattr(sandbox, '__file__', code.co_filename)
    try:
        sys.modules[sandbox.__name__] = sandbox
        exec(code, sandbox.__dict__)
    finally:
        del sys.modules[sandbox.__name__]
        if restore_argv:
            sys.argv = orig_argv


def open_python_script(session, stream, file_name, argv=None):
    """Execute Python script in a ChimeraX context

    Each script is opened in a uniquely named importable sandbox
    (see timeit example above).  And the current ChimeraX session
    is available as a global variable named **session**.

    Parameters
    ----------
    session : a ChimeraX :py:class:`~chimerax.core.session.Session`
    stream : open data stream
    file_name : how to identify the file
    """
    try:
        data = stream.read()
        code = compile(data, stream.name, 'exec')
        _exec_python(session, code, argv)
    except Exception as e:
        session.logger.error(_format_file_exception(stream.name))
        from chimerax.core.errors import UserError
        raise UserError('Error opening python file %s' % stream.name)
    finally:
        stream.close()
    return [], "executed %s" % file_name

def _format_file_exception(file_path):
    '''
    Return formatted exception including only traceback frames
    after the specified code file is reached.
    '''
    import sys
    etype, value, tb = sys.exc_info()
    import traceback
    tb_entries = traceback.extract_tb(tb)
    for i, entry in enumerate(tb_entries):
        if entry.filename == file_path:
            break
    tb_length = len(tb_entries) - i
    limit = None if tb_length == 0 else -tb_length
    msg = ''.join(traceback.format_exception(etype, value, tb, limit = limit))
    return msg

def open_compiled_python_script(session, stream, file_name, argv=None):
    """Execute compiled Python script in a ChimeraX context

    Each script is opened in a uniquely named importable sandbox
    (see timeit example above).  And the current ChimeraX session
    is available as a global variable named **session**.

    Parameters
    ----------
    session : a ChimeraX :py:class:`~chimerax.core.session.Session`
    stream : open data stream
    file_name : how to identify the file
    """
    import pkgutil
    try:
        code = pkgutil.read_code(stream)
        if code is None:
            from .errors import UserError
            raise UserError("Python code was compiled for a different version of Python")
        _exec_python(session, code, argv)
    finally:
        stream.close()
    return [], "executed %s" % file_name


def open_command_script(session, path, file_name):
    """Execute utf-8 file as ChimeraX commands.

    The current directory is changed to the file directory before the commands
    are executed and restored to the previous current directory after the
    commands are executed.

    Parameters
    ----------
    session : a ChimeraX :py:class:`~chimerax.core.session.Session`
    path : path to file to open
    name : how to identify the file
    """
    input = _builtin_open(path, 'rb')

    prev_dir = None
    import os
    dir = os.path.dirname(path)
    if dir:
        prev_dir = os.getcwd()
        os.chdir(dir)

    from .commands import run
    try:
        for line in input.readlines():
            text = line.strip().decode('utf-8', errors='replace')
            run(session, text)
    finally:
        input.close()
        if prev_dir:
            os.chdir(prev_dir)

    return [], "executed %s" % file_name
