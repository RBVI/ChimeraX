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

# -----------------------------------------------------------------------------
# Implementation of select zone subcommand that selects atoms or surfaces near
# other atoms or surfaces.
#
def select_zone(session, near, range, find = None, extend = False, residues = False):
    '''
    Select atoms or surfaces near other atoms or surfaces.
    The distance from a point to a surface is the minimum distance to
    all surface vertices.  Target molecular surfaces are not considered
    because the selection of molecular surfaces mirrors the selection
    of associated atoms and that would produce a confusing result.

    Parameters
    ----------
    near : Objects
       Reference objects
    range : float
       Maximum distance from reference object points
    find : Objects or None
       Target objects to select if they are in range of any reference object.
    extend : bool
       Whether to include all the reference objects in the selection.
    residues : bool
       Whether to select full residues that have an in range atom,
       or just the individual atoms that are in range.
    '''

    na = near.atoms
    ns = surfaces(near.models)
    if len(na) == 0 and len(ns) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No atoms or surfaces specified')

    if find is None:
        from chimerax.core.objects import all_objects
        find = all_objects(session)

    fa = find.atoms
    fs = surfaces(find.models, exclude_molecular_surfaces = True)
    if len(fa) == 0 and len(fs) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No target atoms or surfaces')

    # Remove near stuff.
    if na:
      fa = fa.subtract(na)
    if ns:
      fs = list(set(fs).difference(ns))

    sa, ss = zone_items(na, ns, range, fa, fs, extend, residues)
    select_items(session, sa, ss)
    report_selected(session.logger, sa, ss)

# -----------------------------------------------------------------------------
#
def surfaces(models, exclude_molecular_surfaces = False):
    surfs = []
    from chimerax.atomic import Structure, MolecularSurface
    for s in models:
        if isinstance(s, Structure):
            continue
        if exclude_molecular_surfaces and isinstance(s, MolecularSurface):
            continue
        if not s.empty_drawing():
            surfs.append(s)
    return surfs

# -----------------------------------------------------------------------------
#
def zone_items(na, ns, range, fa, fs, extend = False, residues = False):
    # TODO: Use double precision coordinates and transforms.
    from numpy import float32
    naxyz = na.scene_coords.astype(float32)
    # TODO: Only consider masked geometry.  Handle surface instances.
    nsxyz = [(s.vertices.astype(float32), p.matrix) for s in ns for p in s.get_scene_positions()]
    from chimerax.geometry import Place
    im = Place().matrix
    nxyz = [(naxyz,im)] + nsxyz
    faxyz = fa.scene_coords.astype(float32)
    fsxyz = [(s.vertices.astype(float32), p.matrix) for s in fs for p in s.get_scene_positions()]
    fxyz = [(faxyz,im)] + fsxyz

    from chimerax.geometry import find_close_points_sets
    i1, i2 = find_close_points_sets(nxyz, fxyz, range)

    sel = []
    from numpy import take, compress
    sa = fa.filter(i2[0])

    ss = set(fs for fs,i in zip(fs,i2[1:]) if len(i) > 0)
    if extend:
        sa = sa.merge(na)
        ss.update(ns)

    if residues:
        sa = sa.unique_residues.atoms

    return sa, ss

# -----------------------------------------------------------------------------
#
def select_items(session, sa, ss):
    sel = session.selection
    sel.clear()
    sa.selected = True
    for s in ss:
        s.selected = True

# -----------------------------------------------------------------------------
#
def report_selected(log, sa, ss):
    nsa, nss = len(sa), len(ss)
    if nsa == 0 and nss == 0:
        msg = 'Nothing selected'
    else:
        items = []
        if nsa:
            items.append('%d atoms' % nsa)
        if nss:
            items.append('%d surfaces' % nss)
        msg = 'Selected %s' % ', '.join(items)
    log.status(msg)
    log.info(msg)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg, FloatArg, BoolArg
    desc = CmdDesc(
        required=[('near', ObjectsArg),
                  ('range', FloatArg)],
        optional=[('find', ObjectsArg)],
        keyword=[('extend', BoolArg),
                 ('residues', BoolArg)],
        synopsis='Select atoms or surfaces near other atoms and surfaces'
    )
    register('select zone', desc, select_zone, logger=logger)
