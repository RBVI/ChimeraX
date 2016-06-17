# -----------------------------------------------------------------------------
#
def parse_symmetry(group, center, axis, csys, molecule, cmdname):

    c, a, csys_ca = parse_center_axis(center, axis, csys, cmdname)

    # Handle products of symmetry groups.
    groups = group.split('*')
    tflists = []
    import Matrix as M
    for g in groups:
        tflist, csys_g = group_symmetries(g, c.data(), a.data(), csys, molecule)
        if csys is None:
            csys = csys_g or csys_ca or molecule.openState
        elif csys_g and not csys_g is csys:
            ctf = M.multiply_matrices(M.xform_matrix(csys_g.xform.inverse()),
                                      M.xform_matrix(csys.xform))
            tflist = M.coordinate_transform_list(tflist, ctf)
        tflists.append(tflist)

    tflist = reduce(M.matrix_products, tflists)
    
    return tflist, csys

# -----------------------------------------------------------------------------
# The returned center and axis are in csys coordinates.
#
def parse_center_axis(center, axis, csys, cmdname):

    from Commands import parseCenterArg, parse_axis

    if isinstance(center, (tuple, list)):
        from chimera import Point
        center = Point(*center)
        ccs = csys
    elif center:
        center, ccs = parseCenterArg(center, cmdname)
    else:
        ccs = None

    if isinstance(axis, (tuple, list)):
        from chimera import Vector
        axis = Vector(*axis)
        axis_point = None
        acs = csys
    elif axis:
        axis, axis_point, acs = parse_axis(axis, cmdname)
    else:
        axis_point = None
        acs = None

    if not center and axis_point:
        # Use axis point if no center specified.
        center = axis_point
        ccs = acs

    # If no coordinate system specified use axis or center coord system.
    cs = (ccs or acs)
    if csys is None and cs:
        csys = cs
        xf = cs.xform.inverse()
        if center and not ccs:
            center = xf.apply(center)
        if axis and not acs:
            axis = xf.apply(axis)

    # Convert axis and center to requested coordinate system.
    if csys:
        xf = csys.xform.inverse()
        if center and ccs:
            center = xf.apply(ccs.xform.apply(center))
        if axis and acs:
            axis = xf.apply(acs.xform.apply(axis))

    return center, axis, csys

# -----------------------------------------------------------------------------
#
def group_symmetries(group, center, axis, csys, mol):

    import Symmetry
    from Commands import CommandError

    g0 = group[:1].lower()
    gfields = group.split(',')
    nf = len(gfields)
    recenter = True
    if g0 in ('c', 'd'):
        # Cyclic or dihedral symmetry: C<n>, D<n>
        try:
            n = int(group[1:])
        except ValueError:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        if n < 1:
            raise CommandError('Cn or Dn with n = %d < 1' % (n,))
        if g0 == 'c':
            tflist = Symmetry.cyclic_symmetry_matrices(n)
        else:
            tflist = Symmetry.dihedral_symmetry_matrices(n)
    elif g0 == 'i':
        # Icosahedral symmetry: i[,<orientation>]
        if nf == 1:
            orientation = '222'
        elif nf == 2:
            orientation = gfields[1]
            if not orientation in Symmetry.icosahedral_orientations:
                raise CommandError('Unknown icosahedron orientation "%s"'
                                   % orientation)
        else:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        tflist = Symmetry.icosahedral_symmetry_matrices(orientation)
    elif g0 == 't' and nf <= 2:
        # Tetrahedral symmetry t[,<orientation]
        if nf == 1:
            orientation = '222'
        elif nf == 2:
            orientation = gfields[1]
            if not orientation in Symmetry.tetrahedral_orientations:
                tos = ', '.join(Symmetry.tetrahedral_orientations)
                raise CommandError('Unknown tetrahedral symmetry orientation %s'
                                   ', must be one of %s' % (gfields[1], tos))
        else:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        tflist = Symmetry.tetrahedral_symmetry_matrices(orientation)
    elif g0 == 'o':
        # Octahedral symmetry
        if nf == 1:
            tflist = Symmetry.octahedral_symmetry_matrices()
        else:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
    elif g0 == 'h':
        # Helical symmetry: h,<rise>,<angle>,<n>[,<offset>]
        if nf < 4 or nf > 5:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        try:
            param = [float(f) for f in gfields[1:]]
        except ValueError:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        if len(param) == 3:
            param.append(0.0)
        rise, angle, n, offset = param
        n = int(n)
        tflist = [Symmetry.helical_symmetry_matrix(rise, angle, n = i+offset)
                  for i in range(n)]
    elif gfields[0].lower() == 'shift' or (g0 == 't' and nf >= 3):
        # Translation symmetry: t,<n>,<distance> or t,<n>,<dx>,<dy>,<dz>
        if nf != 3 and nf != 5:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        try:
            param = [float(f) for f in gfields[1:]]
        except ValueError:
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        n = param[0]
        if n != int(n):
            raise CommandError('Invalid symmetry group syntax "%s"' % group)
        n = int(n)
        if nf == 3:
          delta = (0,0,param[1])
        else:
          delta = param[1:]
        tflist = Symmetry.translation_symmetry_matrices(n, delta)
    elif group.lower() == 'biomt':
        # Biological unit
        from Molecule import biological_unit_matrices
        tflist = biological_unit_matrices(mol)
        if len(tflist) == 0:
            raise CommandError('Molecule %s has no biological unit info'
                               % mol.name)
        from Matrix import is_identity_matrix
        if len(tflist) == 1 and is_identity_matrix(tflist[0]):
            from chimera import replyobj
            replyobj.status('Molecule %s is the biological unit' % mol.name)
        if csys is None:
            csys = mol.openState
        else:
            tflist = transform_coordinates(tflist, mol.openState, csys)
        recenter = False
    elif g0 == '#':
        from Commands import models_arg
        if nf == 1:
            mlist = [m for m in models_arg(group) if model_symmetry(m, csys)]
            if len(mlist) == 0:
                raise CommandError('No symmetry for "%s"' % group)
            elif len(mlist) > 1:
                raise CommandError('Multiple models "%s"' % group)
            m = mlist[0]
            tflist = model_symmetry(m, csys)
            if csys is None:
                csys = m.openState
            recenter = False
        elif nf == 2:
            gf0, gf1 = gfields
            mlist = [m for m in models_arg(gf0)
                     if hasattr(m, 'placements') and callable(m.placements)]
            if len(mlist) == 0:
                raise CommandError('No placements for "%s"' % gf0)
            elif len(mlist) > 1:
                raise CommandError('Multiple models with placements "%s"' % gf0)
            m = mlist[0]
            tflist = m.placements(gf1)
            if len(tflist) == 0:
                raise CommandError('No placements "%s" for "%s"' % (gf1, gf0))
            import Molecule as MC, Matrix as M
            c = MC.molecule_center(mol)
            cg = M.apply_matrix(M.xform_matrix(mol.openState.xform), c)
            cm = M.apply_matrix(M.xform_matrix(m.openState.xform.inverse()), cg)
            tflist = make_closest_placement_identity(tflist, cm)
            if csys is None:
                csys = m.openState
            recenter = False
    else:
        raise CommandError('Unknown symmetry group "%s"' % group)

    # Apply center and axis transformation.
    if recenter and (tuple(center) != (0,0,0) or tuple(axis) != (0,0,1)):
        import Matrix as M
        tf = M.multiply_matrices(M.vector_rotation_transform(axis, (0,0,1)),
                                 M.translation_matrix([-c for c in center]))
        tflist = M.coordinate_transform_list(tflist, tf)

    return tflist, csys

# -----------------------------------------------------------------------------
#
def model_symmetry(model, csys):

    from VolumeViewer import Volume
    from chimera import Molecule
    if isinstance(model, Volume):
        tflist = model.data.symmetries
    elif isinstance(model, Molecule):
        from Molecule import biological_unit_matrices
        tflist = biological_unit_matrices(model)
    else:
        tflist = []

    if len(tflist) <= 1:
        return None

    if not csys is None:
        tflist = transform_coordinates(tflist, model.openState, csys)

    return tflist
