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

def functionkey(session, key_name = None, command = None):

    fkc = function_key_commands(session)
    if key_name is None:
        if len(fkc) == 0:
            msg = 'No function keys are assigned'
        else:
            msg = ('Function key commands:\n' + 
                   '\n'.join('F%d: %s' % (key, command) for key, command in fkc.items()))
        session.logger.info(msg)
        return
        
    key_num = None
    if  key_name.startswith('f') or key_name.startswith('F'):
        try:
            key_num = int(key_name[1:])
        except ValueError:
            pass
    if key_num is None:
        from chimerax.core.errors import UserError
        raise UserError('Function key name must be F1, F2, F3..., got "%s"' % key_name)

    if command is None:
        cmd = fkc.get(key_num, 'none assigned')
        msg = 'Function key F%d command: %s' % (key_num, cmd)
        session.logger.status(msg, log = True)
        return

    cmd = None if command == 'none' else command
    _set_function_key_command(session, key_num, cmd)

from chimerax.core.commands import Annotation
class CommandArg(Annotation):
    @staticmethod
    def parse(text, session):
        cmd = text.strip()
        if ((cmd.startswith('"') and cmd.endswith('"')) or
            (cmd.startswith("'") and cmd.endswith("'"))):
            cmd = cmd[1:-1]        # Strip quotes.
        return cmd, text, ''

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, str)
        return value

def register_functionkey_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias, StringArg, WholeRestOfLine
    desc = CmdDesc(
        optional = [('key_name', StringArg),
                    ('command', CommandArg)],
        synopsis = 'Assign a command to a function key'
    )
    register('ui functionkey', desc, functionkey, logger=logger)
    create_alias('functionkey', 'ui functionkey $*', logger=logger)

def function_key_commands(session):
    settings = _function_key_settings(session)
    return settings.function_key_commands

def _set_function_key_command(session, key_num, command):
    settings = _function_key_settings(session)
    fkc = settings.function_key_commands
    d = fkc.copy()	# Need to make a copy or Settings does not save new value.
    if command is None:
        del d[key_num]
    else:
        d[key_num] = command
    settings.function_key_commands = d
    settings.save()

def _function_key_settings(session):
    if not hasattr(session, '_function_key_settings'):
        from chimerax.core.settings import Settings
        class _FunctionKeySettings(Settings):
            EXPLICIT_SAVE = { 'function_key_commands': {}, }
        session._function_key_settings = _FunctionKeySettings(session, "function_keys")
    return session._function_key_settings

def function_key_pressed(session, key_num):
    fkc = function_key_commands(session)
    if key_num in fkc:
        cmd = fkc[key_num]
        from chimerax.core.commands import run
        run(session, cmd)
