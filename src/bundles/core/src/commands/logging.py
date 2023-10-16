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
Routines for logging equivalent commands to actions done
with mouse modes and graphical user interfaces.
"""

def log_equivalent_command(session, command_text):
    """
    Log a command.  This is typically used to show the equivalent
    command for some action (button press, mouse drag, ...) done
    with the grahical user interface.
    """
    from chimerax.core.commands import Command
    from chimerax.core.errors import UserError
    command = Command(session)
    try:
        command.run(command_text, log_only=True)
    except UserError as err:
        session.logger.info(str(err))

def enable_motion_commands(session, enable, frame_skip = 0):
    '''
    Enabling motion commands causes the "motion command" trigger
    to fire to track continuous changes in the scene, typically due
    to mouse drags.  This is used by the meeting command for mirroring
    these actions in multi-person sessions.
    '''
    if not hasattr(session, '_motion_commands_enabled'):
        session.triggers.add_trigger('motion command')
    session._motion_commands_enabled = enable
    session._motion_command_skip = frame_skip

def motion_commands_enabled(session):
    '''
    Whether mouse drag modes should issue equivalent commands using
    the motion_command(session, command) call.
    '''
    if not getattr(session, '_motion_commands_enabled', False):
        return False
    frame_skip = session._motion_command_skip
    if frame_skip == 0:
        return True
    return session.main_view.frame_number % (frame_skip + 1) == 0

def motion_command(session, command_text):
    '''
    Post a motion command used for synchronization in multi-person sessions.
    The command is not executed in the current ChimeraX.
    '''
    if motion_commands_enabled(session):
        session.triggers.activate_trigger('motion command', command_text)

def residues_specifier(objects):
    res = objects.atoms.unique_residues
    specs = []
    for s, cid, cres in res.by_chain:
        rnums = ','.join('%d' % rnum for rnum in cres.numbers)
        if ' ' in cid:
            cspec = "(#%s::chain_id='%s'&:%s)" % (s.id_string, cid, rnums)
        else:
            cspec = '#%s/%s:%s' % (s.id_string, cid, rnums)
        specs.append(cspec)
    spec = ''.join(specs)
    return spec

def camel_case(text):
    words = text.split('_')
    return words[0] + "".join([x.capitalize() for x in words[1:]])

def options_text(options):
    ostring = ' '.join('%s %s' % (camel_case(name), _option_value(value))
                       for name, value in options if value is not None)
    return ostring

def _option_value(value):
    from chimerax.core.colors import Color
    if isinstance(value, (int, float, bool)):
        v = str(value)
    elif isinstance(value, str):
        if ' ' in value:
            v = '"%s"' % str
        else:
            v = str
    elif isinstance(value, Color):
        if hasattr(value, 'color_name'):
            v = value.color_name
        else:
            rgba = value.uint8x4()
            if rgba[3] == 255:
                v = '#%02x%02x%02x' % tuple(rgba[:3])
            else:
                v = '#%02x%02x%02x%02x' % tuple(rgba)
    else:
        raise ValueError('Unknown value type %s' % str(type(value)))
    return v
