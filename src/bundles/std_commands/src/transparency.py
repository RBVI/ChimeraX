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

def transparency(session, objects, percent, what=None, target=None):
    """Set transparency of atoms, ribbons, surfaces, ....

    Parameters
    ----------
    objects : Objects or None
      Which objects to set transparency.
    percent : float
      Percent transparent from 0 completely opaque, to 100 completely transparent.
    target : string
      Characters indicating what to make transparent:
      a = atoms, b = bonds, p = pseudobonds, c = cartoon, r = cartoon, s = surfaces, A = all
    """
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from .color import get_targets
    target, _ = get_targets(target, what, 's')

    alpha = int(2.56 * (100 - percent))
    alpha = min(255, max(0, alpha))    # 0-255 range

    what = []

    if 'a' in target or 's' in target:
        atoms = objects.atoms

    if 'a' in target:
        # atoms
        c = atoms.colors
        c[:, 3] = alpha
        atoms.colors = c
        what.append('%d atoms' % len(atoms))

    if 'b' in target:
        # bonds
        bonds = objects.bonds
        if bonds:
            c = bonds.colors
            c[:, 3] = alpha
            bonds.colors = c
            what.append('%d bonds' % len(bonds))

    if 'p' in target:
        # pseudobonds
        bonds = objects.pseudobonds
        if bonds:
            c = bonds.colors
            c[:, 3] = alpha
            bonds.colors = c
            what.append('%d pseudobonds' % len(bonds))

    if 's' in target:
        surfs = _set_surface_transparency(atoms, objects.models, session, alpha)
        what.append('%d surfaces' % len(surfs))

    if 'c' in target or 'r' in target:
        residues = objects.residues
        c = residues.ribbon_colors
        c[:, 3] = alpha
        residues.ribbon_colors = c
        what.append('%d residues' % len(residues))

    if 'm' in target:
        models = _set_model_transparency(objects.models, session, alpha)
        what.append('%d models' % len(models))

    if not what:
        what.append('nothing')

    from chimerax.core.commands import commas
    session.logger.status('Set transparency of %s' % commas(what, 'and'))

def _set_surface_transparency(atoms, models, session, alpha):

    # Handle surfaces for specified atoms
    from chimerax import atomic
    surfs = atomic.surfaces_with_atoms(atoms)
    for s in surfs:
        vcolors = s.vertex_colors
        amask = s.atoms.mask(atoms)
        all_atoms = amask.all()
        if all_atoms:
            c = s.colors
            c[:, 3] = alpha
            s.colors = c
            if vcolors is None:
                continue

        if vcolors is None:
            from numpy import empty, uint8
            vcolors = empty((len(s.vertices), 4), uint8)
            vcolors[:] = s.color
        v2a = s.vertex_to_atom_map()
        if v2a is None:
            if amask.all():
                v = slice(len(vcolors))
            else:
                session.logger.info('No atom associations for surface #%s' % s.id_string)
                continue
        else:
            v = amask[v2a]
        vcolors[v, 3] = alpha
        s.vertex_colors = vcolors

    # Handle surface models specified without specifying atoms
    from chimerax.atomic import MolecularSurface
    from chimerax.map import Volume
    from chimerax.core.models import Surface
    osurfs = []
    for s in models:
        if isinstance(s, MolecularSurface):
            if not s in surfs:
                osurfs.append(s)
        elif isinstance(s, Volume) or isinstance(s, Surface):
            osurfs.append(s)
    for s in osurfs:
        s.set_transparency(alpha)
    surfs.extend(osurfs)
            
    return surfs

def _set_model_transparency(models, session, alpha):
    for m in models:
        r,g,b,a = m.color
        m.color = (r,g,b,alpha)
    return models

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, Or, ObjectsArg, EmptyArg, FloatArg
    from chimerax.core.commands import ListOf, EnumOf
    from .color import TargetArg
    from .color import WHAT_TARGETS
    what_arg = ListOf(EnumOf((*WHAT_TARGETS.keys(),)))
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('percent', FloatArg)],
                   optional=[('what', what_arg)],
                   keyword=[('target', TargetArg)],
                   synopsis="change object transparency")
    register('transparency', desc, transparency, logger=logger)
