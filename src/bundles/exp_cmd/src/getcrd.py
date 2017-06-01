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

from chimerax.core.commands import register, CmdDesc, EnumOf, AtomSpecArg

def initialize(command_name, logger):
    register("getcrd", getcrd_desc, getcrd, logger=logger)

def getcrd(session, atoms=None, coordinate_system='scene'):
    from chimerax.core.core_settings import settings
    if atoms is None:
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    atoms = results.atoms
    msgs = []
    if coordinate_system == 'scene':
        coords = atoms.scene_coords
    elif coordinate_system == 'model':
        coords = atoms.coords
    elif coordinate_system == 'screen':
        s2c = session.main_view.camera.position.inverse()
        coords = atoms.scene_coords
        s2c.move(coords)
    save = settings.atomspec_contents
    settings.atomspec_contents = "command-line specifier"
    for i, a in enumerate(atoms):
        c = coords[i]
        msgs.append("Atom %s %.3f %.3f %.3f" % (a.atomspec(), c[0], c[1], c[2]))
    settings.atomspec_contents = save
    session.logger.info('\n'.join(msgs))
getcrd_desc = CmdDesc(required=[("atoms", AtomSpecArg),],
                      optional=[("coordinate_system", EnumOf(('scene', 'model', 'screen')))],
                      synopsis='report atomic coordinates')
