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

def palette_list(session):
    '''
    List available palette names.
    '''
    logger = session.logger
    _list(logger, session.user_colormaps, "user-defined")
    from .. import colors
    _list(logger, colors.BuiltinColormaps, "built-in")

def _list(logger, colormaps, kind):
    names = []
    for name, cm in colormaps.items():
        if cm.name is not None:
            name = cm.name
        names.append(name)
    if len(names) == 0:
        return
    elif len(names) > 2:
        parts = [", ".join(names[:-1]), names[-1]]
    else:
        parts = names
    logger.info(kind.capitalize() + " palettes: " + " and ".join(parts))

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc
    desc = CmdDesc(synopsis="list available palette names")
    register('palette list', desc, palette_list, logger=session.logger)
