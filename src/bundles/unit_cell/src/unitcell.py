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

# ---------------------------------------------------------------------------
#
def transforms(molecule, num_cells = (1,1,1), offset = (0,0,0), origin = (0,0,0),
               use_sym_from_file = True, use_spacegroup = True,
               use_ncs = True, pack = True):

    from chimerax import pdb_matrices as pm, crystal

    from chimerax.geometry import Places, Place
    sm = Places([])
    if use_sym_from_file:
        sm = pm.crystal_symmetries(molecule, use_space_group_table = False)
    if len(sm) == 0 and use_spacegroup:
        sm = pm.space_group_symmetries(molecule)

    mm = Places()
    if use_ncs:
        mm = pm.noncrystal_symmetries(molecule)

    if len(sm) > 0:
        tflist = sm * mm
    else:
        tflist = mm

    # Adjust transforms so centers of models are in unit cell box
    cp = pm.unit_cell_parameters(molecule)
    uc = cp[:6] if cp else None
    if pack and uc:
        mc = _molecule_center(molecule)
        tflist = crystal.pack_unit_cell(uc, origin, mc, tflist)

    # Make multiple unit cells
    nc = num_cells
    if nc != (1,1,1) and uc:
        # Compute origin.
        oc = tuple((((o+n-1)%n)-(n-1)) for o,n in zip(offset, nc))
        tflist = crystal.unit_cell_translations(uc, oc, nc, tflist)

    return tflist

# -----------------------------------------------------------------------------
#
def place_molecule_copies(m, parent_model, tflist):

    clist = []
    for i,tf in enumerate(tflist):
        name = m.name + (' #%d' % (i+1))
        c = _find_model_by_name(m.session, name, parent = parent_model)
        if c is None:
            c = m.copy(name)
            c._unit_cell_copy = m
            _transform_atom_positions(c.atoms, tf)
            clist.append(c)
        else:
            _transform_atom_positions(c.atoms, tf, m.atoms)
        c.scene_position = m.scene_position
    m.session.models.add(clist, parent = parent_model)

# ---------------------------------------------------------------------------
#
def copies_group_model(m, create = True):
    gname = m.name + ' unit cell'
    gm = _find_model_by_name(m.session, gname)
    if gm is None and create:
        from chimerax.core.models import Model
        gm = Model(gname, m.session)
        gm.model_panel_show_expanded = False
        m.session.models.add([gm])
    return gm

# -----------------------------------------------------------------------------
#
def showing_copies(m):
    return _find_model_by_name(m.session, copies_group_model(m, create=False)) is not None

# -----------------------------------------------------------------------------
# Move atoms in molecule coordinate system using a 3 by 4 matrix.
#
def _transform_atom_positions(atoms, tf, from_atoms = None):
    from_coords = atoms.scene_coords if from_atoms is None else from_atoms.scene_coords
    atoms.scene_coords = tf * from_coords
    
# -----------------------------------------------------------------------------
#
def remove_extra_copies(m, parent_model, nkeep):

  clist = []
  while True:
    name = m.name + (' #%d' % (len(clist)+nkeep+1))
    c = _find_model_by_name(m.session, name, parent = parent_model)
    if c is None:
      break
    clist.append(c)
  m.session.models.close(clist)

# -----------------------------------------------------------------------------
#
def _find_model_by_name(session, name, parent = None):

  mlist = session.models.list() if parent is None else parent.child_models()
  for m in mlist:
    if m.name == name:
      return m
  return None

# -----------------------------------------------------------------------------
#
def _molecule_center(m):
    return m.atoms.scene_coords.mean(axis = 0)

# ---------------------------------------------------------------------------
#
def show_outline_model(m, origin, outline_model = None):
    from chimerax import pdb_matrices as pm, crystal
    cp = pm.unit_cell_parameters(m)
    if cp is None:
        return
    a, b, c, alpha, beta, gamma, space_group, zvalue = cp

    axes = crystal.unit_cell_axes(a, b, c, alpha, beta, gamma)
    mc = _molecule_center(m)
    origin = crystal.cell_origin(origin, axes, mc)
    color = (1,1,1)                     # white

    if outline_model is None:
        name = _outline_model_name(m)
        s = _new_outline_box(m.session, name, origin, axes, color)
        s.scene_position = m.scene_position
    else:
        _update_outline_box(outline_model, origin, axes, color)

# ---------------------------------------------------------------------------
#
def outline_model(molecule):
    return _find_model_by_name(molecule.session, _outline_model_name(molecule))

# ---------------------------------------------------------------------------
#
def _outline_model_name(molecule):
    return molecule.name + ' unit cell outline'

# -----------------------------------------------------------------------------
#
def _new_outline_box(session, name, origin, axes, rgb):
    from chimerax.core.models import Surface
    surface_model = s = Surface(name, session)
    s.display_style = s.Mesh
    s.use_lighting = False
    s.casts_shadows = False
    s.pickable = False
    s.outline_box = True
    _update_outline_box(s, origin, axes, rgb)
    session.models.add([s])
    return s

# -----------------------------------------------------------------------------
#
def _update_outline_box(surface_model, origin, axes, rgb):

    a0, a1, a2 = axes
    from numpy import array, float32, int32, uint8
    c000 = array(origin, float32)
    c100 = c000 + a0
    c010 = c000 + a1
    c001 = c000 + a2
    c110 = c100 + a1
    c101 = c100 + a2
    c011 = c010 + a2
    c111 = c011 + a0
    va = array((c000, c001, c010, c011, c100, c101, c110, c111), float32)
    ta = array(((0,4,5), (5,1,0), (0,2,6), (6,4,0),
                (0,1,3), (3,2,0), (7,3,1), (1,5,7),
                (7,6,2), (2,3,7), (7,5,4), (4,6,7)), int32)

    b = 2 + 1    # Bit mask, edges are bits 4,2,1
    hide_diagonals = array((b,b,b,b,b,b,b,b,b,b,b,b), uint8)

    # Replace the geometry of the surface
    surface_model.set_geometry(va, None, ta)
    surface_model.edge_mask = hide_diagonals

    rgba = tuple(rgb) + (1,)
    surface_model.color = tuple(int(255*r) for r in rgba)

# -----------------------------------------------------------------------------
#
def unit_cell_info(m):
    from chimerax import pdb_matrices as pm
    cp = pm.unit_cell_parameters(m)
    if cp:
        a, b, c, alpha, beta, gamma, space_group, zvalue = cp
        cs = '%7.3f %7.3f %7.3f' % (a,b,c) if not None in (a,b,c) else ''
        if None in (alpha,beta,gamma):
            ca = ''
        else:
            import math
            radians_to_degrees = 180 / math.pi
            alpha_deg = radians_to_degrees * alpha
            beta_deg = radians_to_degrees * beta
            gamma_deg = radians_to_degrees * gamma
            ca = '%6.2f %6.2f %6.2f' % (alpha_deg,beta_deg,gamma_deg)
        if space_group is None:
            sg = sgsc = ''
        else:
            sg = space_group
            from chimerax import crystal
            sgm = crystal.space_group_matrices(space_group, a, b, c,
                                               alpha, beta, gamma)
            sgsc = '%d' % len(sgm) if sgm else '0'
    else:
        sg = cs = ca = sgsc = ''

    sm = pm.crystal_symmetries(m, use_space_group_table = False)
    mm = pm.noncrystal_symmetries(m, add_identity = False)
    if len(sm) == 0 and len(mm) == 0 and sg == '' and cs == '' and ca == '' and sgsc == '':
        info = ''
    else:
        info = '\n'.join(['Space group: ' + sg,
                          'Cell size: ' + cs,
                          'Cell angles: ' + ca,
                          'Space group symmetries: ' + sgsc,
                          'Crystal symmetries in file: %d' % len(sm),
                          'Non-crystal symmetries in file: %d' % len(mm)])
    return info
