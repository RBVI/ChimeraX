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

from chimerax.core.commands import register, CmdDesc, BoolArg, AtomSpecArg

def initialize(command_name):
    register("getcrd", getcrd_desc, getcrd)

def getcrd(session, spec=None, scene=True):
    from chimerax.core.core_settings import settings
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    atoms = results.atoms
    msgs = []
    if scene:
        coords = atoms.scene_coords
    else:
        coords = atoms.coords
    save = settings.atomspec_contents
    settings.atomspec_contents = "command-line specifier"
    for i, a in enumerate(atoms):
        c = coords[i]
        msgs.append("Atom %s %.3f %.3f %.3f" % (a.atomspec(), c[0], c[1], c[2]))
    settings.atomspec_contents = save
    session.logger.info('\n'.join(msgs))
getcrd_desc = CmdDesc(required=[("spec", AtomSpecArg),],
                      keyword=[("scene", BoolArg),],
                      synopsis='report atomic coordinates')
