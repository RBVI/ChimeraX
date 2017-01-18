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

# -----------------------------------------------------------------------------
# Implementation of command "zonesel" that selects atoms or surfaces near
# other atoms or surfaces.
#
def zonesel(session, near, range, find = None, extend = False):
    '''
    Select atoms or surfaces near other atoms or surfaces.

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
    '''

    na = near.atoms
    from ..atomic import Structure
    # TODO: Include volume models as surfaces.
    ns = [s for s in near.models if not isinstance(s, Structure) and not s.empty_drawing()]
    if len(na) == 0 and len(ns) == 0:
        from ..errors import UserError
        raise UserError('No atoms or surfaces specified')

    if find is None:
        from . import all_objects
        find = all_objects(session)

    fa = find.atoms
    fs = [s for s in find.models if not isinstance(s, Structure) and not s.empty_drawing()]
    if len(fa) == 0 and len(fs) == 0:
        from ..errors import UserError
        raise UserError('No target atoms or surfaces')

    # Remove near stuff.
    if na:
      fa = fa.subtract(na)
    if ns:
      fs = list(set(fs).difference(ns))

    # TODO: Use double precision coordinates and transforms.
    from numpy import float32
    naxyz = na.scene_coords.astype(float32)
    # TODO: Only consider masked geometry.  Handle surface instances.
    nsxyz = [(s.vertices.astype(float32), p.matrix) for s in ns for p in s.get_scene_positions()]
    from ..geometry import Place
    im = Place().matrix
    nxyz = [(naxyz,im)] + nsxyz
    faxyz = fa.scene_coords.astype(float32)
    fsxyz = [(s.vertices.astype(float32), p.matrix) for s in fs for p in s.get_scene_positions()]
    fxyz = [(faxyz,im)] + fsxyz

    from ..geometry import find_close_points_sets
    i1, i2 = find_close_points_sets(nxyz, fxyz, range)

    sel = []
    from numpy import take, compress
    sa = fa.filter(i2[0])

    ss = set(fs for fs,i in zip(fs,i2[1:]) if len(i) > 0)
    if extend:
        sa = sa.merge(na)
        ss.update(ns)

    sel = session.selection
    sel.clear()
    sa.selected = True
    for s in ss:
        s.selected = True
                    
# -----------------------------------------------------------------------------
#
def register_command(session):
    from .cli import CmdDesc, register, ObjectsArg, FloatArg, NoArg
    desc = CmdDesc(
        required=[('near', ObjectsArg),
                  ('range', FloatArg)],
        optional=[('find', ObjectsArg)],
        keyword=[('extend', NoArg)],
        synopsis='Select atoms or surfaces near other atoms and surfaces'
    )
    register('zonesel', desc, zonesel)
