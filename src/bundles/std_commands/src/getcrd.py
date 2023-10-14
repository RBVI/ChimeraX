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

from chimerax.core.commands import register, CmdDesc, EnumOf, AtomSpecArg, atomspec

def register_command(logger):
    register("getcrd", getcrd_desc, getcrd, logger=logger)

def getcrd(session, atoms=None, coordinate_system='scene'):
    from chimerax.atomic.settings import settings
    if atoms is None:
        atoms = atomspec.everything(session)
    results = atoms.evaluate(session)
    atoms = results.atoms
    if not atoms:
        from chimerax.core.errors import UserError
        raise UserError("No atoms specified")
    msgs = []
    if coordinate_system == 'scene':
        coords = atoms.scene_coords
    elif coordinate_system == 'model':
        coords = atoms.coords
    elif coordinate_system == 'screen':
        s2c = session.main_view.camera.position.inverse()
        coords = atoms.scene_coords
        s2c.transform_points(coords, in_place=True)
    save = settings.atomspec_contents
    settings.atomspec_contents = "command"
    for i, a in enumerate(atoms):
        c = coords[i]
        msgs.append("Atom %s %.3f %.3f %.3f" % (a.atomspec, c[0], c[1], c[2]))
    settings.atomspec_contents = save
    session.logger.info('\n'.join(msgs))
    return coords

getcrd_desc = CmdDesc(required=[("atoms", AtomSpecArg),],
                      optional=[("coordinate_system", EnumOf(('scene', 'model', 'screen')))],
                      synopsis='report atomic coordinates')
