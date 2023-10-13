# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
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
        from chimerax.core.errors import UserError
        if probably_chimera1_session(e):
           raise UserError(chimera1_session_message)
        session.logger.error(_format_file_exception(stream.name))
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


def open_command_script(session, path, file_name, log = True, for_each_file = None):
    """Execute utf-8 file as ChimeraX commands.

    The current directory is changed to the file directory before the commands
    are executed and restored to the previous current directory after the
    commands are executed.

    Parameters
    ----------
    session : a ChimeraX :py:class:`~chimerax.core.session.Session`
    path : path to file to open
    file_name : how to identify the file
    log : whether to log each script command
    for_each_file : data file paths, iterate opening each file followed by the script which has $file replaced by filename with suffix stripped.
    """
    if for_each_file is not None:
        return apply_command_script_to_files(session, path, file_name, for_each_file, log = log)
    
    input = _builtin_open(path, 'rb')
    commands = [cmd.strip().decode('utf-8', errors='replace') for cmd in input.readlines()]
    input.close()

    from os.path import dirname
    _run_commands(session, commands, directory = dirname(path), log = log)

    return [], "executed %s" % file_name

def _run_commands(session, commands, directory = None, log = True):
    if directory:
        import os
        prev_dir = os.getcwd()
        os.chdir(directory)

    from .commands import run
    try:
        for cmd in commands:
            run(session, cmd, log=log)
    finally:
        if directory:
            os.chdir(prev_dir)

def apply_command_script_to_files(session, path, script_name, for_each_file, log = True):
    input = _builtin_open(path, 'rb')
    commands = [cmd.strip().decode('utf-8', errors='replace') for cmd in input.readlines()]
    input.close()

    paths = []
    from glob import glob
    from os.path import expanduser
    for path in for_each_file:
        paths.extend(glob(expanduser(path)))

    from os.path import basename, dirname, splitext
    from .commands import run
    for i, data_path in enumerate(paths):
        run(session, 'close', log = log)
        run(session, 'open %s' % data_path, log = log)
        session.logger.status('Executing script %s on file %s (%d of %d)'
                              % (basename(script_name), basename(data_path), i+1, len(paths)))
        fprefix = splitext(basename(data_path))[0]
        cmds = [cmd.replace('$file', fprefix) for cmd in commands]
        _run_commands(session, cmds, directory = dirname(data_path), log = log)

    return [], "executed %s on %d data files" % (script_name, len(paths))

def probably_chimera1_session(evalue):
    if type(evalue) != ModuleNotFoundError:
        return False
    if 'cPickle' not in str(evalue):
        return False
    from traceback import format_exception
    import sys
    formatted = format_exception(*sys.exc_info())
    if len(formatted) > 1 and ' line 1,' in formatted[-2]:
        return True
    return False

chimera1_session_message = """\
ChimeraX cannot open a regular Chimera session.  An exporter from Chimera
to ChimeraX is being worked on but only handles molecules and molecular surfaces
(not volumes) at this time.  If that is sufficient, use the latest Chimera
daily build and its File->Export Scene menu item, and change the resulting
dialog's "File Type" to ChimeraX."""
