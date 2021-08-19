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
#
def mmcif_unit_cell_matrices(molecule, pack = None, group = False):

  slist = mmcif_crystal_symmetry_matrices(molecule)
  mlist = mmcif_ncs_matrices(molecule)

  cp = mmcif_unit_cell_parameters(molecule)
  uc = cp[:6] if cp else None
  from chimerax import crystal
  smlist = crystal.unit_cell_matrices(slist, mlist, uc, pack, group)
  return smlist

# -----------------------------------------------------------------------------
#
def mmcif_unit_cell_parameters(molecule):

  from chimerax import mmcif
  cell_table = mmcif.get_mmcif_tables_from_metadata(molecule, ['cell'])[0]
  if cell_table is None:
    return None

  pnames = ('length_a', 'length_b', 'length_c',
            'angle_alpha', 'angle_beta', 'angle_gamma')
  params = cell_table.fields(pnames)[0]

  # Cell parameters can have uncertainty in parentheses at end.  Ick.
  try:
    cell = [float(a) for a in params]
  except ValueError:
    return None

  from math import pi
  cell = cell[:3] + [a*pi/180 for a in cell[3:]]  # convert degrees to radians

  z_pdb = cell_table.fields(['z_pdb'])[0][0]
  if z_pdb:
    try:
      z = int(z_pdb)
    except ValueError:
      z = None

  symmetry_table = mmcif.get_mmcif_tables_from_metadata(molecule, ['symmetry'])[0]
  if symmetry_table:
    sg = symmetry_table.fields(['space_group_name_h-m'], missing_value=None)[0][0]
  else:
    sg = None

  return cell + [sg, z]

# -----------------------------------------------------------------------------
#
def mmcif_crystal_symmetry_matrices(molecule):

  # TODO: No crystal symmetry matrices in example file 1bbt.cif.  Probably
  #       have to use space group name to lookup symmetries.
  cp = mmcif_unit_cell_parameters(molecule)
  if cp:
    a, b, c, alpha, beta, gamma, space_group, zvalue = cp
    from chimerax import crystal
    sgt = crystal.space_group_matrices(space_group, a, b, c, alpha, beta, gamma)
  else:
    from chimerax.geometry import Places
    sgt = Places([])

  # Handle crystal symmetry origin not equal to atom coordinate origin
  origin = mmcif_crystal_origin(molecule)
  if origin != (0,0,0):
    shift = [-x for x in origin]
    from chimerax.geometry import translation
    sgt = sgt.transform_coordinates(translation(shift))
    
  return sgt

# -----------------------------------------------------------------------------
#
def mmcif_ncs_matrices(molecule, include_given = True):

  from chimerax.geometry import Places
  from chimerax import mmcif
  struct_ncs_oper_table = mmcif.get_mmcif_tables_from_metadata(molecule, ['struct_ncs_oper'])[0]
  if struct_ncs_oper_table is None:
    return Places([])

  entries = struct_ncs_oper_table.fields(['id', 'code'] + matrix_field_names)
  tflist = []
  for fields in entries:
    if include_given or fields[1] != 'given':
      id = int(fields[0])
      tf = mmcif_matrix(fields[2:14])
      tflist.append((id, tf))
  tflist.sort()
  return Places([tf for id, tf in tflist])

# -----------------------------------------------------------------------------
#
matrix_field_names = [
  'matrix[1][1]', 'matrix[1][2]', 'matrix[1][3]',
  'matrix[2][1]', 'matrix[2][2]', 'matrix[2][3]',
  'matrix[3][1]', 'matrix[3][2]', 'matrix[3][3]',
  'vector[1]', 'vector[2]', 'vector[3]']

def mmcif_matrix(matrix_values):

  m11, m12, m13, m21, m22, m23, m31, m32, m33, v1, v2, v3 = \
       [float(v) for v in matrix_values]
  tf = ((m11, m12, m13, v1),
        (m21, m22, m23, v2),
        (m31, m32, m33, v3))
  from chimerax.geometry import Place
  return Place(tf)

# -----------------------------------------------------------------------------
#
def mmcif_biounit_matrices(molecule):

  # TODO: mmCIF file lists multiple assemblies, not just a single biological
  # unit like PDB format files.  This code returns matrices of all the numbered
  # operators in pdbx_struct_oper_list.  It would be better to choose the first
  # mmcif assembly but that can apply different operators to different chains.
    
  from chimerax.geometry import Places

  from chimerax import mmcif
  pdbx_struct_oper_list_table = mmcif.get_mmcif_tables_from_metadata(molecule,
                                                                     ['pdbx_struct_oper_list'])[0]
  if pdbx_struct_oper_list_table is None:
    return Places([])

  entries = pdbx_struct_oper_list_table.fields(['id'] + matrix_field_names)
  tflist = []
  for fields in entries:
    try:
      i = int(fields[0])
    except ValueError:
      continue
    tflist.append((i, mmcif_matrix(fields[1:13])))
  tflist.sort()
  return Places([tf for i, tf in tflist])

# -----------------------------------------------------------------------------
# PDB entries allow the origin of the crystal unit cell symmetries
# to be different from the origin of atom coordinates.  It is rare, e.g. PDB 1WAP.
# Given in mmcif _atom_sites.fract_transf_vector
#
def mmcif_crystal_origin(molecule):
  from chimerax import mmcif
  atom_sites_table = mmcif.get_mmcif_tables_from_metadata(molecule, ['atom_sites'])[0]
  if atom_sites_table is None:
    return (0,0,0)

  pnames = ['fract_transf_vector[%d]' % i for i in (1,2,3)]
  fields = atom_sites_table.fields(pnames)[0]
  try:
    forigin = [float(x) for x in fields]
  except:
    return (0,0,0)

  # Convert fractional unit cell coordinates to atom coordinates
  cp = mmcif_unit_cell_parameters(molecule)
  if cp is None:
    return (0,0,0)
  
  fx,fy,fz = forigin
  a, b, c, alpha, beta, gamma, space_group, zvalue = cp
  from chimerax import crystal
  ax,ay,az = crystal.unit_cell_axes(a, b, c, alpha, beta, gamma)
  from chimerax.geometry import linear_combination
  origin = tuple(linear_combination(fx,ax,fy,ay,fz,az))

  return origin
