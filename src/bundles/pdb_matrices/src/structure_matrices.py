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
# Return cell size and angles (a, b, c, alpha, beta, gamma, space_group, z).
# Angles are in radians.
#
def unit_cell_parameters(molecule):

    from chimerax import pdb_matrices as pm
    fmt = metadata_type(molecule)
    if fmt == 'pdb':
        cp = pm.pdb_crystal_parameters(molecule.metadata)
    elif fmt == 'mmcif':
        cp = pm.mmcif_unit_cell_parameters(molecule)
    elif fmt == 'cif':
        cp = pm.cif_unit_cell_parameters(molecule)
    else:
        cp = None
    return cp

# -----------------------------------------------------------------------------
# To get all the transformations needed to build the unit cell, multiply all
# crystallographic symmetry matrices by all non-crystallographic symmetry
# matrices.
#
# The pack argument can be set to a pair of points
# (molecule-center, unit-cell-origin) and the unit cell transforms will be
# translated to put all molecule centers in the unit cell box.
#
def unit_cell_matrices(molecule, pack = None, group = False, cells = (1,1,1)):

    if tuple(cells) == (1,1,1):
        from chimerax import pdb_matrices as pm
        fmt = metadata_type(molecule)
        if fmt == 'pdb':
            m = pm.pdb_unit_cell_matrices(molecule.metadata, pack, group)
        elif fmt == 'mmcif':
            m = pm.mmcif_unit_cell_matrices(molecule, pack, group)
        elif fmt == 'cif':
            m = pm.cif_unit_cell_matrices(molecule, pack, group)
        else:
            from chimerax.geometry import Places
            m = Places([])
    else:
        cp = unit_cell_parameters(molecule)
        if cp is None:
            from chimerax.geometry import Places
            m = Places([])
        else:
            from chimerax import crystal
            a, b, c, alpha, beta, gamma = cp[:6]
            cell_axes = crystal.unit_cell_axes(a, b, c, alpha, beta, gamma)
            cranges = [(int(2-c)/2,int(c)/2)for c in cells]
            mlist = crystal.translation_matrices(cell_axes, cranges)
            clist = unit_cell_matrices(molecule, pack = pack, group = group)
            m = crystal.matrix_products(mlist, clist, group)
        
    return m

# -----------------------------------------------------------------------------
#
def crystal_symmetries(molecule, use_space_group_table = True):

    from chimerax import pdb_matrices as pm
    fmt = metadata_type(molecule)
    if fmt == 'pdb':
        s = pm.pdb_smtry_matrices(molecule.metadata)
        # Handle crystal symmetry origin not equal to atom coordinate origin
        origin = pm.pdb_crystal_origin(molecule.metadata)
        if origin != (0,0,0):
            shift = [-x for x in origin]
            from chimerax.geometry import translation
            s = s.transform_coordinates(translation(shift))
    elif fmt == 'mmcif':
        s = pm.mmcif_crystal_symmetry_matrices(molecule)
    elif fmt == 'cif':
        s = pm.cif_crystal_symmetry_matrices(molecule)
    else:
        from chimerax.geometry import Places
        s = Places([])
    if len(s) == 0 and use_space_group_table:
        s = space_group_symmetries(molecule)
        
    return s

# -----------------------------------------------------------------------------
# In PDB files the SCALE1, SCALE2, SCALE3 remark records can indicate that
# the center of spacegroup symmetry is not 0,0,0 in atom coordinates.
# This is rare, seems to only to be older entries, for example, 1WAP.
# The PDB SMTRY remarks don't account for the different origin.
#
def crystal_symmetry_origin(molecule):
    from chimerax import pdb_matrices as pm
    fmt = metadata_type(molecule)
    if fmt == 'pdb':
        origin = pm.pdb_crystal_origin(molecule.metadata)
    elif fmt == 'mmcif':
        origin = pm.mmcif_crystal_origin(molecule)
    else:
        origin = (0,0,0)
    return origin

# -----------------------------------------------------------------------------
#
def noncrystal_symmetries(molecule, add_identity = True):

    fmt = metadata_type(molecule)
    from chimerax import pdb_matrices as pm
    if fmt == 'pdb':
        s = pm.pdb_mtrix_matrices(molecule.metadata, add_identity = False)
    elif fmt == 'mmcif':
        s = pm.mmcif_ncs_matrices(molecule, include_given = False)
    elif fmt == 'cif':
        s = pm.cif_ncs_matrices(molecule)
    else:
        from chimerax.geometry import Places
        s = Places([])
    if add_identity:
        if not [m for m in s if m.is_identity()]:
            from chimerax.geometry import Place, Places
            s = Places([Place()] + list(s))
    return s

# -----------------------------------------------------------------------------
#
def biological_unit_matrices(molecule):

    fmt = metadata_type(molecule)
    from chimerax import pdb_matrices as pm
    if fmt == 'pdb':
        s = pm.pdb_biomt_matrices(molecule.metadata)
    elif fmt == 'mmcif':
        s = pm.mmcif_biounit_matrices(molecule)
    else:
        from chimerax.geometry import Places
        s = Places([])
    return s

# -----------------------------------------------------------------------------
#
def metadata_type(molecule):

    mdata = molecule.metadata
    if 'REMARK' in mdata or 'CRYST1' in mdata:
        mtype = 'pdb'
    elif getattr(molecule, 'is_mmcif', False) or has_mmcif_tables(mdata):
        mtype = 'mmcif'
    else:
        mtype = None
    # TODO: Add small-molecule CIF format if a reader is added to ChimeraX.
    return mtype
    
# -----------------------------------------------------------------------------
#
def has_mmcif_tables(metadata, table_names = ['atom_sites','cell','symmetry','struct_ncs_oper','pdbx_struct_oper_list']):
    for name in table_names:
        if name in metadata:
            return True
    return False

# -----------------------------------------------------------------------------
#
def space_group_symmetries(molecule):

    cp = unit_cell_parameters(molecule)
    if cp:
        a, b, c, alpha, beta, gamma, space_group, zvalue = cp
        from chimerax import crystal
        sgt = crystal.space_group_matrices(space_group, a, b, c,
                                           alpha, beta, gamma)
    else:
        from chimerax.geometry import Places
        sgt = Places([])
    return sgt
