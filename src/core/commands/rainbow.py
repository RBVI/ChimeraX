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

def rainbow(session, objects, level='residues', target=None, transparency=None,
            palette=None, halfbond=None):
    '''
    Color residues or chains by sequence using a color map.
    Arguments are the same as for the color command.
    '''
    from .color import color
    color(session, objects, target=target, transparency=transparency,
          sequential=level, palette=palette, halfbond=halfbond)

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ColormapArg, ObjectsArg
    from . import EmptyArg, Or, EnumOf, StringArg, TupleOf, FloatArg, BoolArg
    from .color import _SequentialLevels
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('level', EnumOf(_SequentialLevels))],
                   keyword=[('target', StringArg),
                            ('transparency', FloatArg),
                            ('palette', ColormapArg),
                            ('halfbond', BoolArg)],
                   url='help:user/commands/color.html#rainbow',
                   synopsis="color residues and chains sequentially")
    register('rainbow', desc, rainbow)
