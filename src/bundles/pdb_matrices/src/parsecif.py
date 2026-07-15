# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def cif_unit_cell_matrices(molecule, pack = None, group = False):

  slist = cif_crystal_symmetry_matrices(molecule)
  mlist = cif_ncs_matrices(molecule)

  cp = cif_unit_cell_parameters(molecule)
  uc = cp[:6] if cp else None
  from chimerax import crystal
  smlist = crystal.unit_cell_matrices(slist, mlist, uc, pack, group)
  return smlist

# -----------------------------------------------------------------------------
#
def cif_unit_cell_parameters(molecule):
  from . import parsemmcif
  return parsemmcif.mmcif_unit_cell_parameters(molecule, 'formula_units_z')

# -----------------------------------------------------------------------------
#
def cif_crystal_symmetry_matrices(molecule):
  
  from chimerax.geometry import Places
  from chimerax import mmcif
  from chimerax.mmcif import TableMissingFieldsError

  equiv, space_group = mmcif.get_mmcif_tables_from_metadata(molecule, ['symmetry_equiv', 'space_group_symop'])
  if equiv is None and space_group is None:
    return Places([])

  if equiv is not None:
    # pnames = ('pos_site_id' a.k.a. 'id', 'pos_as_xyz')
    pnames = ('pos_as_xyz',)
    try:
      params = equiv.fields(pnames)
    except TableMissingFieldsError:
      return Places([])
    sops = [p[0] for p in params]
  else:
    pnames = ('operation_xyz',)
    try:
      params = space_group.fields(pnames)
    except TableMissingFieldsError:
      return Places([])
    sops = [p[0] for p in params]

  from chimerax.crystal.space_groups import parse_symop
  ftflist = Places([parse_symop(sop.upper().replace(' ','')) for sop in sops])

  # Convert from fractional coordinates to xyz.
  uc = cif_unit_cell_parameters(molecule)[:6]
  if None in uc:
    return Places([])
  from chimerax import crystal
  u2r = crystal.unit_cell_to_xyz_matrix(*uc)
  tflist = ftflist.transform_coordinates(u2r.inverse())
  
  return tflist

# -----------------------------------------------------------------------------
#
def cif_ncs_matrices(molecule):

  from chimerax.geometry import Places
  return Places([])     # TODO: Find out if and how CIF files support NCS symmetry
