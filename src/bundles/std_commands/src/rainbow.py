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
            palette=None):
    '''
    Color residues or chains by sequence using a color map.
    Arguments are the same as for the color command.
    '''
    from .color import color_sequential
    color_sequential(session, objects, target=target, transparency=transparency,
                     level=level, palette=palette, undo_name="rainbow")

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, ColormapArg, ObjectsArg
    from chimerax.core.commands import EmptyArg, Or, EnumOf, StringArg, TupleOf, FloatArg, BoolArg
    from .color import _SequentialLevels
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg))],
                   optional=[('level', EnumOf(_SequentialLevels))],
                   keyword=[('target', StringArg),
                            ('transparency', FloatArg),
                            ('palette', ColormapArg),
                   ],
                   url='help:user/commands/color.html#rainbow',
                   synopsis="color residues and chains sequentially")
    register('rainbow', desc, rainbow, logger=logger)
