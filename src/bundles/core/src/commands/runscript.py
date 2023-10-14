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

def runscript(session, script_file, *, args=None):
    """Execute a Python or ChimeraX command script with arguments

    Parameters
    ----------
    script_file : string
        Path to Python script file
    args : string
        Optional string containing the arguments to pass to the script
    """

    is_python = script_file.endswith('.py')
    is_commands = script_file.endswith('.cxc')
    if not is_python and not is_commands:
        from chimerax.core.errors import UserError
        raise UserError(f'Script "{script_file}" must have suffix .py or .cxc')
    
    import shlex
    from ..scripting import open_python_script
    argv = [script_file]
    if args is not None:
        argv += shlex.split(args)
    with session.in_script:
        if is_python:
            open_python_script(session, open(script_file, 'rb'), script_file, argv=argv)
        elif is_commands:
            run_command_script(session, script_file, argv[1:])
    return []

def run_command_script(session, path, args):
    with open(path, 'rb') as f:
        lines = [cmd.strip().decode('utf-8', errors='replace') for cmd in f.readlines()]

    commands = _replace_arguments(lines, args)

    from chimerax.core.scripting import _run_commands
    _run_commands(session, commands)

def _replace_arguments(lines, args):
    repl = [(f'${i+1}',arg) for i, arg in enumerate(args)]
    repl.reverse()  # Handle arg $10 before arg $1
    cmds = []
    for cmd in lines:
        for var,value in repl:
            cmd = cmd.replace(var,value)
        cmds.append(cmd)
    return cmds
