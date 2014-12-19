# -----------------------------------------------------------------------------
# Compute center of mass of a map for the region above a specifie contour level.
# Returns center map index coordinates.
#
def volume_center_of_mass(v, level = None):

    if level is None:
        # Use lowest displayed contour level.
        level = min(v.surface_levels)

    # Get 3-d array of map values.
    m = v.data.full_matrix()

    # Find indices of map values above displayed threshold.
    kji = (m >= level).nonzero()

    # Compute total mass above threshold.
    msum = m[kji].sum()

    # Compute mass-weighted center
    center = [(i*m[kji]).sum()/msum for i in kji]
    center.reverse()        # k,j,i -> i,j,k index order

    return center

# -----------------------------------------------------------------------------
# Compute center of mass of atoms.
#
def atoms_center_of_mass(atoms, cxf = None):

    import Molecule
    xyz = Molecule.atom_positions(atoms, cxf)
    from numpy import array, float32, dot
    weights = array([a.element.mass for a in atoms], float32)
    w = weights.sum()
    c = tuple(dot(weights,xyz)/w)
    return c

# -----------------------------------------------------------------------------
#
def atoms_center_model_name(atoms):

    mols = set(a.molecule for a in atoms)
    if len(mols) == 1:
        m0 = mols.pop()
        name = m0.name + ' center'
        if len(atoms) != len(m0.atoms):
            name += ' %d atoms' % (len(atoms),)
    else:
        name = 'center %d atoms' % (len(atoms),)
    return name

# -----------------------------------------------------------------------------
#
def place_marker(xyz, msys, color, radius, model_name = None, model_id = None):

    # Locate specified marker model
    mset = None
    import VolumePath as VP
    if not model_id is None:
        msets = VP.marker_sets(model_id)
        if len(msets) >= 1:
            mset = msets[0]

    if mset is None:
        # Create a new marker model
        mname = msys.name + ' center' if model_name is None else model_name
        mset = VP.Marker_Set(mname)
        mset.marker_molecule(model_id)

    mos = mset.marker_molecule().openState
    if mos != msys.openState:
        # Transform from volume to marker model coordinates.
        import Matrix
        xyz = Matrix.xform_xyz(xyz, msys.openState.xform, mos.xform)

    # Place marker at center position.
    mset.place_marker(xyz, color, radius)
