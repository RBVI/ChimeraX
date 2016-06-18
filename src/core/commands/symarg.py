# vim: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
#
from . import Annotation
class SymmetryArg(Annotation):
    '''Symmetry specification, e.g. C3 or D7'''
    name = 'symmetry'

    @staticmethod
    def parse(text, session):
        from . import next_token
        group, atext, rest = next_token(text)
        return Symmetry(group), atext, rest

# -----------------------------------------------------------------------------
#
class Symmetry:
    def __init__(self, group):
        self.group = group
    def positions(self, center = None, axis = None, coord_sys = None,
                  molecule = None):
        '''
        center is a Center object.
        axis is an Axis object.
        coord_sys is a Place object mapping center/axis coordinates to scene coordinates,
        molecule is for biomt symmetries.
        Returned positions is a Places object in scene coordinates.
        '''
        if center:
            c = center.scene_coordinates(coord_sys)
        elif axis and axis.base_point() is not None:
            c = axis.base_point()
        elif coord_sys:
            c = coord_sys.origin()
        else:
            c = None
        if axis:
            a = axis.scene_coordinates(coord_sys)
        elif coord_sys:
            a = coord_sys.z_axis()
        else:
            a = None
        return parse_symmetry(self.group, c, a, molecule)

# Molecule arg is used for biomt symmetry.
def parse_symmetry(group, center = None, axis = None, molecule = None):

    # Handle products of symmetry groups.
    groups = group.split('*')
    from ..geometry import Places
    ops = Places()
    for g in groups:
        ops = ops * group_symmetries(g, molecule)

    # Apply center and axis transformation.
    if center is not None or axis is not None:
        from ..geometry import Place, vector_rotation, translation
        tf = Place()
        if center is not None and tuple(center) != (0,0,0):
            tf = translation([-c for c in center])
        if axis is not None and tuple(axis) != (0,0,1):
            tf = vector_rotation(axis, (0,0,1)) * tf
        if not tf.is_identity():
            ops = ops.transform_coordinates(tf)
    return ops

# -----------------------------------------------------------------------------
#
def group_symmetries(group, molecule):

    from .. import geometry
    from ..errors import UserError

    g0 = group[:1].lower()
    gfields = group.split(',')
    nf = len(gfields)
    recenter = True
    if g0 in ('c', 'd'):
        # Cyclic or dihedral symmetry: C<n>, D<n>
        try:
            n = int(group[1:])
        except ValueError:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        if n < 1:
            raise UserError('Cn or Dn with n = %d < 1' % (n,))
        if g0 == 'c':
            tflist = geometry.cyclic_symmetry_matrices(n)
        else:
            tflist = geometry.dihedral_symmetry_matrices(n)
    elif g0 == 'i':
        # Icosahedral symmetry: i[,<orientation>]
        if nf == 1:
            orientation = '222'
        elif nf == 2:
            orientation = gfields[1]
            if not orientation in geometry.icosahedral_orientations:
                raise UserError('Unknown icosahedron orientation "%s"'
                                   % orientation)
        else:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        tflist = geometry.icosahedral_symmetry_matrices(orientation)
    elif g0 == 't' and nf <= 2:
        # Tetrahedral symmetry t[,<orientation]
        if nf == 1:
            orientation = '222'
        elif nf == 2:
            orientation = gfields[1]
            if not orientation in geometry.tetrahedral_orientations:
                tos = ', '.join(geometry.tetrahedral_orientations)
                raise UserError('Unknown tetrahedral symmetry orientation %s'
                                   ', must be one of %s' % (gfields[1], tos))
        else:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        tflist = geometry.tetrahedral_symmetry_matrices(orientation)
    elif g0 == 'o':
        # Octahedral symmetry
        if nf == 1:
            tflist = geometry.octahedral_symmetry_matrices()
        else:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
    elif g0 == 'h':
        # Helical symmetry: h,<rise>,<angle>,<n>[,<offset>]
        if nf < 4 or nf > 5:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        try:
            param = [float(f) for f in gfields[1:]]
        except ValueError:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        if len(param) == 3:
            param.append(0.0)
        rise, angle, n, offset = param
        n = int(n)
        tflist = [geometry.helical_symmetry_matrix(rise, angle, n = i+offset)
                  for i in range(n)]
    elif gfields[0].lower() == 'shift' or (g0 == 't' and nf >= 3):
        # Translation symmetry: t,<n>,<distance> or t,<n>,<dx>,<dy>,<dz>
        if nf != 3 and nf != 5:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        try:
            param = [float(f) for f in gfields[1:]]
        except ValueError:
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        n = param[0]
        if n != int(n):
            raise UserError('Invalid symmetry group syntax "%s"' % group)
        n = int(n)
        if nf == 3:
          delta = (0,0,param[1])
        else:
          delta = param[1:]
        tflist = geometry.translation_symmetry_matrices(n, delta)
    elif group.lower() == 'biomt':
        # Biological unit
        from ..atomic import biological_unit_matrices
        tflist = biological_unit_matrices(molecule)
        if len(tflist) == 0:
            raise UserError('Molecule %s has no biological unit info' % molecule.name)
        if len(tflist) == 1 and tflist[0].is_identity():
            log = molecule.session.logger
            log.status('Molecule %s is the biological unit' % molecule.name)
        tflist = tflist.transform_coordinates(molecule.position.inverse())
        recenter = False
    elif g0 == '#':
        from . import ModelsAarg
        if nf == 1:
            models = ModelsArg.parse(group, session)[0]
            mlist = [m for m in models if model_symmetry(models)]
            if len(mlist) == 0:
                raise UserError('No symmetry for "%s"' % group)
            elif len(mlist) > 1:
                raise UserError('Multiple models "%s"' % group)
            m = mlist[0]
            tflist = model_symmetry(m)
            recenter = False
        elif nf == 2:
            gf0, gf1 = gfields
            models = ModelsArg.parse(gf0, session)[0]
            mlist = [m for m in models
                     if hasattr(m, 'placements') and callable(m.placements)]
            if len(mlist) == 0:
                raise UserError('No placements for "%s"' % gf0)
            elif len(mlist) > 1:
                raise UserError('Multiple models with placements "%s"' % gf0)
            m = mlist[0]
            tflist = m.placements(gf1)
            if len(tflist) == 0:
                raise UserError('No placements "%s" for "%s"' % (gf1, gf0))
            c = mol.atoms.coords.mean(axis = 0)
            cg = molecule.position * c
            cm = m.position.inverse() * cg
            tflist = make_closest_placement_identity(tflist, cm)
            recenter = False
    else:
        raise UserError('Unknown symmetry group "%s"' % group)

    return tflist

# -----------------------------------------------------------------------------
#
def model_symmetry(model):

    from ..map import Volume
    from ..atomic import Structure
    if isinstance(model, Volume):
        tflist = model.data.symmetries
    elif isinstance(model, Structure):
        from ..atomic import biological_unit_matrices
        tflist = biological_unit_matrices(model)
    else:
        tflist = []

    if len(tflist) <= 1:
        return None

    return tflist

# -----------------------------------------------------------------------------
# Find the transform that maps (0,0,0) closest to the molecule center and
# multiply all transforms by the inverse of that transform.  This chooses
# the best placement for the current molecule position and makes all other
# placements relative to that one.
#
def make_closest_placement_identity(tflist, center):

    d = tflist.array()[:,:,3] - center
    d2 = (d*d).sum(axis = 1)
    i = d2.argmin()
    tfinv = tflist[i].inverse()
    rtflist = [tf*tfinv for tf in tflist]
    from ..geometry import Place, Places
    rtflist[i] = Place()
    return Places(rtflist)
