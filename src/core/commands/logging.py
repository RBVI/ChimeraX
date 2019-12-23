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

def residues_specifier(objects):
    res = objects.atoms.unique_residues
    spec = ''.join('#%s/%s:%s' % (s.id_string, cid, ','.join('%d' % rnum for rnum in cres.numbers))
                   for s, cid, cres in res.by_chain)
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

        
