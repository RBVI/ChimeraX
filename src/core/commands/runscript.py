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

def runscript(session, script_file, *, args=None):
    """Execute a Python script with arguments

    Parameters
    ----------
    script_file : string
        Path to Python script file
    args : string
        Optional string containing the arguments to pass to the script
    """

    import shlex
    from ..scripting import open_python_script
    argv = [script_file]
    if args is not None:
        argv += shlex.split(args)
    with session.in_script:
        open_python_script(session, open(script_file, 'rb'), script_file, argv=argv)
    return []
