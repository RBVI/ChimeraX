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
# Command to create standard geometric shapes as surfaces primarily for
# masking volume data.
#
#   Syntax: shape cylinder|ellipsoid|icosahedron|rectangle|ribbon|sphere|tube
#               [radius <r>|<rx,ry,rz>]
#               [divisions <d>]
#               [height <h>]
#               [center <x,y,z>|spec]
#               [rotation <ax,ay,az,a>]
#               [qrotation <qx,qy,qz,qw>]
#               [caps true|false]
#               [coordinateSystem <modelid>]
#               [color <cname>]
#               [mesh true|false]
#               [sphereFactor <f>]
#               [orientation <o>]
#               [lattice <h,k>]
#               [slab <width>|<d1,d2>]
#               [segmentSubdivisions <n>]
#               [followBonds true|false]
#               [bandLength <l>]
#               [name <name>]
#               [model_id <modelid>]
#
# where orientation applies only to icosahedron and can be one of
# 222, 2n5, n25, 2n3 or any of those with r appended.  The lattice parameter
# is two non-negative integers separated by a comma and controls the
# layout of hexagons for an icosahedron.
#

from chimerax.core.errors import UserError as CommandError

# -----------------------------------------------------------------------------
#
def shape_box_path(session, atoms, width = 1.0, twist = 0.0, color = (190,190,190,255),
                   center = None, rotation = None, qrotation = None,
                   coordinate_system = None, mesh = False, slab = None,
                   report_cuts = False, cut_scale = 1.0,
                   name = 'box path', model_id = None):

    if len(atoms) < 2:
        raise CommandError('Must specify at least 2 atoms, got %d' % len(atoms))

    points = atoms.scene_coords

    from .boxpath import box_path
    varray, tarray = box_path(points, width, twist)

    if report_cuts:
        from .boxpath import cut_distances
        cuts = cut_distances(varray)
        lines = '\n'.join(['\t'.join(['%6.2f'%(d*cut_scale,) for d in cut])
                           for cut in cuts])
        session.logger.info('Box cuts for %d segments\n' % len(cuts) + lines + '\n')

    p = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name)
    return p

# -----------------------------------------------------------------------------
#
def shape_cone(session, radius = 1.0, top_radius = 0.0, height = 1.0,
               from_point = None, to_point = None, axis = None,
               center = None, rotation = None, qrotation = None,
               coordinate_system = None,
               divisions = 72, color = (190,190,190,255),
               mesh = False,
               caps = True, slab = None,
               name = 'cone', model_id = None):

    hcr = _line_orientation(from_point, to_point, axis, height)
    if hcr is not None:
        height, center, rotation = hcr

    r = max(radius, top_radius)
    nz, nc = _cylinder_divisions(r, height, divisions)
    varray, tarray = cylinder_geometry(r, height, nz, nc, caps)

    rh = r*height
    if rh > 0:
        f0 = radius/rh
        f1 = top_radius/rh
        z = varray[:,2]
        f = f0*(0.5*height-z) + f1*(0.5*height+z)
        varray[:,0] *= f
        varray[:,1] *= f

    s = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name)
    return s

# -----------------------------------------------------------------------------
#
def shape_cylinder(session, radius = 1.0, height = 1.0,
                   from_point = None, to_point = None, axis = None,
                   center = None, rotation = None, qrotation = None,
                   coordinate_system = None,
                   divisions = 72, color = (190,190,190,255),
                   mesh = False,
                   caps = True, slab = None,
                   name = 'cylinder', model_id = None):

    hcr = _line_orientation(from_point, to_point, axis, height)
    if hcr is not None:
        height, center, rotation = hcr
        
    nz, nc = _cylinder_divisions(radius, height, divisions)
    varray, tarray = cylinder_geometry(radius, height, nz, nc, caps)

    s = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name)
    return s

# -----------------------------------------------------------------------------
#
def _line_orientation(from_point, to_point, axis, height):
    if from_point is None and to_point is None and axis is None:
        return None

    c,h,r = None, height, None

    from chimerax.core.commands import Axis, Center
    if isinstance(axis, Axis):
        axis = axis.scene_coordinates()
    if isinstance(from_point, Center):
        from_point = from_point.scene_coordinates()
    if isinstance(to_point, Center):
        to_point = to_point.scene_coordinates()
        
    from chimerax.geometry import vector_rotation, norm
    if axis is not None:
        r = vector_rotation((0,0,1), axis)
    else:
        from numpy import array, float32
        axis = array((0,0,1), float32)
        
    if from_point is not None and to_point is not None:
        c = 0.5 * (from_point + to_point)
        v = to_point - from_point
        r = vector_rotation((0,0,1), v)
        h = norm(v)
    elif from_point is not None and to_point is None:
        c = from_point + 0.5*height*axis
    elif from_point is None and to_point is not None:
        c = to_point - 0.5*height*axis
        
    return h, c, r

# -----------------------------------------------------------------------------
#
def _cylinder_divisions(radius, height, divisions):

    from math import ceil, pi, sqrt
    nc = max(3, int(ceil(divisions)))
    nz = max(2, int(ceil(nc * height / (sqrt(3)*pi*radius))))
    return nz, nc

# -----------------------------------------------------------------------------
#
def cylinder_geometry(radius, height, nz, nc, caps):

    varray, tarray = tube_geometry(nz, nc)
    varray[:,0] *= radius
    varray[:,1] *= radius
    varray[:,2] *= height
   
    if not caps:
        return varray, tarray


    # Duplicate end circle vertices so they can have different normals.
    #
    # NOTE: resize does not zero out the array on resize! It's fine
    # here, we fill in the array. But we must make sure not to allow
    # trash values through in future refactors.
    from numpy import resize
    vc = varray.shape[0]
    varray = resize(varray, (vc+2*nc+2,3))
    varray[vc,:] = (0,0,-0.5*height)
    varray[vc+1:vc+1+nc,:] = varray[:nc,:]
    varray[vc+nc+1,:] = (0,0,0.5*height)
    varray[vc+nc+2:,:] = varray[(nz-1)*nc:nz*nc,:]

    tc = tarray.shape[0]
    tarray = resize(tarray, (tc+2*nc,3))
    for i in range(nc):
        tarray[tc+i,:] = (vc,vc+1+(i+1)%nc,vc+1+i)
        tarray[tc+nc+i,:] = (vc+nc+1,vc+nc+2+i,vc+nc+2+(i+1)%nc)

    return varray, tarray

# -----------------------------------------------------------------------------
# Build a hexagonal lattice tube
#
def tube_geometry(nz, nc):

    from numpy import zeros, single as floatc, arange, cos, sin, intc, pi
    vc = nz*nc
    tc = (nz-1)*nc*2
    varray = zeros((vc,3), floatc)
    tarray = zeros((tc,3), intc)

    v = varray.reshape((nz,nc,3))
    angles = (2*pi/nc)*arange(nc)
    v[::2,:,0] = cos(angles)
    v[::2,:,1] = sin(angles)
    angles += pi/nc
    v[1::2,:,0] = cos(angles)
    v[1::2,:,1] = sin(angles)
    for z in range(nz):
        v[z,:,2] = z/(nz-1) - 0.5
    t = tarray.reshape((nz-1,nc,6))
    c = arange(nc)
    c1 = (c+1)%nc
    t[:,:,0] = t[1::2,:,3] = c
    t[::2,:,1] = t[::2,:,3] = t[1::2,:,1] = c1
    t[::2,:,4] = t[1::2,:,2] = t[1::2,:,4] = c1+nc
    t[::2,:,2] = t[:,:,5] = c+nc
    for z in range(1,nz-1):
        t[z,:,:] += z*nc

    return varray, tarray

# -----------------------------------------------------------------------------
#
def shape_icosahedron(session, radius = 1.0, center = None, rotation = None,
                      qrotation = None, coordinate_system = None,
                      divisions = 72,
                      color = (190,190,190,255), mesh = None,
                      sphere_factor = 0.0, orientation = '222', lattice = None,
                      slab = None, name = 'icosahedron', model_id = None):

    from chimerax.geometry.icosahedron import coordinate_system_names as csnames
    if orientation not in csnames:
        raise CommandError('Unknown orientation "%s", use %s'
                           % (orientation, ', '.join(csnames)))
    if lattice is not None and mesh is None:
        mesh = True

    if lattice is None:
        varray, tarray = icosahedral_geometry(radius, divisions,
                                              sphere_factor, orientation)
        edge_mask = None
    else:
        hk = lattice
        varray, tarray, edge_mask = hk_icosahedral_geometry(radius, hk,
                                                            sphere_factor,
                                                            orientation)
    s = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name, edge_mask = edge_mask)
    return s

# -----------------------------------------------------------------------------
#
def icosahedral_geometry(radius, divisions, sphere_factor = 0,
                         orientation = '222'):

    from math import pi, log
    from chimerax.geometry.icosahedron import icosahedron_edge_length, icosahedron_triangulation
    d = divisions * (icosahedron_edge_length() / (2*pi))
    subdivision_levels = max(0, int(round(log(d)/log(2))))
    varray, tarray = icosahedron_triangulation(radius, subdivision_levels,
                                               sphere_factor, orientation)
    return varray, tarray

# -----------------------------------------------------------------------------
#
def hk_icosahedral_geometry(radius, hk, sphere_factor = 0, orientation = '222'):

    h,k = hk
    from chimerax.hkcage import cage
    varray, tarray, hex_edges = cage.hk_icosahedron_lattice(h, k, radius, orientation)
    cage.interpolate_with_sphere(varray, radius, sphere_factor)
    return varray, tarray, hex_edges

# -----------------------------------------------------------------------------
#
def shape_rectangle(session, width = 1.0, height = 1.0,
                    center = None, rotation = None, qrotation = None,
                    coordinate_system = None,
                    width_divisions = None, height_divisions = None,
                    divisions = 10,
                    color = (190,190,190,255),
                    mesh = False,
                    slab = None, name = 'rectangle', model_id = None):

    if width_divisions is None:
        width_divisions = divisions
    if height_divisions is None:
        height_divisions = divisions
    from math import ceil
    nw = max(2,int(ceil(width_divisions+1)))
    nh = max(2,int(ceil(height_divisions+1)))
    
    varray, tarray = rectangle_geometry(width, height, nw, nh)

    s = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name, sharp_slab = True)
    return s

# -----------------------------------------------------------------------------
#
def rectangle_geometry(width, height, nw, nh):

    from numpy import empty, single as floatc, arange, intc
    vc = nw*nh
    tc = (nw-1)*(nh-1)*2
    varray = empty((vc,3), floatc)
    tarray = empty((tc,3), intc)

    v = varray.reshape((nh,nw,3))
    ws = width/(nw-1)
    w = arange(nw)*ws - 0.5*width
    for k in range(nh):
        v[k,:,0] = w
    hs = height/(nh-1)
    h = arange(nh)*hs - 0.5*height
    for k in range(nw):
        v[:,k,1] = h
    v[:,:,2] = 0

    t = tarray.reshape((nh-1,nw-1,6))
    c = arange(nw-1)
    t[:,:,0] = t[:,:,3] = c
    t[:,:,2] = c+nw
    t[:,:,1] = t[:,:,5] = c+(nw+1)
    t[:,:,4] = c+1
    for k in range(1,nh-1):
        t[k,:,:] += k*nw

    return varray, tarray

# -----------------------------------------------------------------------------
#
def shape_ribbon(session, atoms, follow_bonds = False,
                 width = 1.0, height = 0.1, yaxis = None, twist = 0,
                 divisions = 15, segment_subdivisions = 10,
                 color = None, band_length = 0.0,
                 mesh = None,
                 name = 'ribbon', model_id = None):

    if len(atoms) == 0:
        raise CommandError('No atoms specified')

    from chimerax.surface.tube import ribbon_through_atoms
    va,na,ta,ca = ribbon_through_atoms(atoms, width, yaxis, twist, band_length,
                                       segment_subdivisions, divisions,
                                       follow_bonds, color)

    if va is None:
        return None

    if height != 0:
        from chimerax.mask.depthmask import slab_surface
        va, na, ta = slab_surface(va, ta, na, (-0.5*height, 0.5*height), sharp_edges = True)
        from numpy import concatenate
        ca = concatenate((ca,ca,ca,ca))

    s = _surface_model(session, model_id, name)
    s.set_geometry(va, na, ta)
    s.vertex_colors = ca
    if mesh or (mesh is None and width == 0):
        s.display_style = s.Mesh
    _add_surface(s)

    return s

# -----------------------------------------------------------------------------
# Makes sphere or ellipsoid if radius is given as 3 values.
#
def shape_sphere(session, radius = 1.0, center = None, rotation = None,
                 qrotation = None, coordinate_system = None,
                 divisions = 72,
                 color = (190,190,190,255), mesh = False,
                 slab = None, name = None, model_id = None):

    ntri = _sphere_triangles(divisions)
    from chimerax.geometry.sphere import sphere_triangulation
    varray, tarray = sphere_triangulation(ntri)
    
    if isinstance(radius, (float, int)):
        varray *= radius
        sphere = True
    elif isinstance(radius, (list, tuple)):
        for a in range(3):
            varray[:,a] *= radius[a]
        sphere = (radius[1] == radius[0] and radius[2] == radius[0])
    else:
        # TODO: Need to handle numpy arrays.
        raise CommandError('shape sphere: radius is not a float, int, list or tuple, got "%s" type %s'
                           % (radius, repr(radius)))

    from chimerax.core.commands import Center
    if isinstance(center, Center):
        center = center.scene_coordinates()
        
    if name is None:
        name = 'sphere' if sphere else 'ellipsoid'

    s = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name)
    return s

# -----------------------------------------------------------------------------
#
def _sphere_triangles(divisions):
    # Assume equilateral triangles with divisions giving number
    # of triangle edges along equator of sphere.
    from math import sqrt, pi
    n = divisions*divisions*4/(sqrt(3)*pi)
    n = max(4, int(n))
    n = 2 * n // 2  # Require even.
    return n

# -----------------------------------------------------------------------------
#
def _show_surface(session, varray, tarray, color, mesh,
                  center, rotation, qrotation, coordinate_system,
                  slab, model_id, shape_name, edge_mask = None, sharp_slab = False):
        
    if center is not None or rotation is not None or qrotation is not None:
        from chimerax.geometry import Place, translation
        tf = Place()
        if rotation is not None:
            tf = rotation * tf
        if qrotation is not None:
            tf = qrotation * tf
        if center is not None:
            from chimerax.core.commands import Center
            if isinstance(center, Center):
                center = center.scene_coordinates()
            tf = translation(center) * tf
        varray = tf.transform_points(varray)
        
    from chimerax.surface import calculate_vertex_normals
    narray = calculate_vertex_normals(varray, tarray)
    
    if slab is not None:
        from chimerax.mask.depthmask import slab_surface
        varray, narray, tarray = slab_surface(varray, tarray, narray, slab,
                                              sharp_edges = sharp_slab)

    s = _surface_model(session, model_id, shape_name, coordinate_system)
    s.set_geometry(varray, narray, tarray)
    if color is not None:
        s.color = color
    if mesh:
        s.display_style = s.Mesh
    if edge_mask is not None:
        s.edge_mask = edge_mask    # Hide spokes of hexagons.
    _add_surface(s)

    return s

# -----------------------------------------------------------------------------
#
def _surface_model(session, model_id, shape_name, position = None):

    s = None if model_id is None else _find_surface_model(session, model_id)
    if s is None:
        from chimerax.core.models import Surface
        s = Surface(shape_name, session)
        if model_id is not None:
            s.id = model_id
        if position is not None:
            s.position = position
        s.SESSION_SAVE_DRAWING = True
        s.clip_cap = True			# Cap surface when clipped
    return s

# -----------------------------------------------------------------------------
#
def _find_surface_model(session, model_id):

    from chimerax.core.models import Surface
    mlist = session.models.list(type = Surface, model_id = model_id)
    return mlist[0] if len(mlist) == 1 else None

# -----------------------------------------------------------------------------
#
def _add_surface(surface):
    session = surface.session
    models = session.models
    if models.have_model(surface):
        # Reusing existing model.
        return

    # Check if request model id is already being used.
    model_id = surface.id
    if model_id:
        if models.have_id(model_id):
            from chimerax.core.errors import UserError
            id_string = '.'.join('%d'%i for i in model_id)
            raise UserError('Model id #%s already in use' % id_string)

        # If parent models don't exist create grouping models.
        p = None
        for i in range(1,len(model_id)):
            if not models.have_id(model_id[:i]):
                from chimerax.core.models import Model
                m = Model('shape', session)
                m.id = model_id[:i]
                models.add([m], parent = p)
                p = m

    # Add surface.
    models.add([surface])

# -----------------------------------------------------------------------------
#
def shape_triangle(session, atoms = None, point = None,
                   color = (190,190,190,255), mesh = False,
                   center = None, rotation = None, qrotation = None, coordinate_system = None,
                   divisions = 1,
                   slab = None, name = 'triangle', model_id = None):

    points = point  # List of points.  Name is point so user command repeated option name is "point"
    if atoms is not None:
        if len(atoms) != 3:
            raise CommandError('shape triangle: Must specify 3 atoms, got %d' % len(atoms))
        if (center is not None or rotation is not None or
            qrotation is not None or coordinate_system is not None):
            raise CommandError('shape triangle: Cannot use center, rotation, qrotation, '
                               'or coordinateSystem options if atom positions are used.')
        vertices = atoms.scene_coords
    elif points is not None:
        if len(points) != 3:
            raise CommandError('shape triangle: Must specify 3 points, got %d' % len(points))
        from chimerax.core.commands import Center
        vertices = [(p.scene_coordinates() if isinstance(p,Center) else p) for p in points]
    else:
        # Equilateral triangle centered at origin, edge length 1.
        from math import sqrt
        vertices = ((-0.5,-sqrt(3)/6,0),(0.5,-sqrt(3)/6,0),(0,sqrt(3)/3,0))
        
    from numpy import array, float32, int32
    varray = array(vertices, float32)
    tarray = array(((0,1,2),), int32)

    div = 1
    while divisions > div:
        from chimerax.surface import subdivide_triangles
        varray, tarray = subdivide_triangles(varray, tarray)
        div *= 2
        
    s = _show_surface(session, varray, tarray, color, mesh,
                      center, rotation, qrotation, coordinate_system,
                      slab, model_id, name, sharp_slab = True)
    return s

# -----------------------------------------------------------------------------
#
def shape_tube(session, atoms, radius = 1.0, band_length = 0.0, follow_bonds = False,
               divisions = 15, segment_subdivisions = 10,
               color = None, mesh = None,
               name = 'tube', model_id = None):

    if len(atoms) == 0:
        raise CommandError('No atoms specified')

    from chimerax.surface.tube import tube_through_atoms
    va,na,ta,ca = tube_through_atoms(atoms, radius, band_length,
                                     segment_subdivisions, divisions,
                                     follow_bonds, color)
    if va is None:
        return None

    s = _surface_model(session, model_id, name)
    s.set_geometry(va, na, ta)
    s.vertex_colors = ca
    if mesh or (mesh is None and radius == 0):
        s.display_style = s.Mesh
    _add_surface(s)

    return s

# -------------------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation, AnnotationError
class SlabArg(Annotation):
    name = '1 or 2 floats'
    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import FloatsArg
        s, text, rest = FloatsArg.parse(text, session)
        if len(s) == 1:
            v = (-s[0]/2, s[0]/2)
        elif len(s) == 2:
            v = s
        else:
            raise AnnotationError('Slab value must be 1 or 2 floats, got %d from %s' % (len(s), text))
        return v, text, rest

# -------------------------------------------------------------------------------------
#
class AxisAngleArg(Annotation):
    name = 'rotation given by axis and angle (degrees) ax,ay,az,angle'
    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import FloatsArg
        aa, text, rest = FloatsArg.parse(text, session)
        if len(aa) != 4:
            raise AnnotationError('Axis-angle must be 4 comma-separated floats, got %d from %s'
                                  % (len(aa), text))
        from chimerax.geometry import rotation
        v = rotation(aa[:3], aa[3])
        return v, text, rest

# -------------------------------------------------------------------------------------
#
class QuaternionArg(Annotation):
    name = 'rotation given by quaternion qw,qx,qy,qz'
    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import FloatsArg
        q, text, rest = FloatsArg.parse(text, session)
        if len(q) != 4:
            raise AnnotationError('Quaternion must be 4 comma-separated floats, got %d from %s'
                                  % (len(q), text))
        from chimerax.geometry import quaternion_rotation
        v = quaternion_rotation(q)
        return v, text, rest

# -------------------------------------------------------------------------------------
#
def register_shape_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, RepeatOf
    from chimerax.core.commands import Color8Arg, BoolArg, StringArg
    from chimerax.core.commands import PositiveIntArg, Int2Arg, FloatArg, Float3Arg
    from chimerax.core.commands import CenterArg, AxisArg, CoordSysArg, ModelIdArg
    from chimerax.map import Float1or3Arg
    from chimerax.atomic import AtomsArg

    base_args = [('color', Color8Arg),
                 ('mesh', BoolArg),
                 ('name', StringArg),
                 ('model_id', ModelIdArg)]

    basic_args = [('divisions', PositiveIntArg)] + base_args

    position_args = [('center', CenterArg),
                     ('rotation', AxisAngleArg),
                     ('qrotation', QuaternionArg),
                     ('coordinate_system', CoordSysArg),
                     ('slab', SlabArg)]

    common_args = position_args + basic_args
    
    # Box beam path
    boxpath_desc = CmdDesc(required = [('atoms', AtomsArg)],
                           keyword = [('width', FloatArg),
                                      ('twist', FloatArg),
                                      ('report_cuts', BoolArg),
                                      ('cut_scale', FloatArg)] + base_args,
                           synopsis = 'create a box beam model')
    register('shape boxPath', boxpath_desc, shape_box_path, logger=logger)

    # Cone
    cone_desc = CmdDesc(keyword = [('radius', FloatArg),
                                   ('top_radius', FloatArg),
                                   ('height', FloatArg),
                                   ('from_point', CenterArg),
                                   ('to_point', CenterArg),
                                   ('axis', AxisArg),
                                   ('caps', BoolArg)] + common_args,
                        synopsis = 'create a cone model')
    register('shape cone', cone_desc, shape_cone, logger=logger)

    # Cylinder
    cylinder_desc = CmdDesc(keyword = [('radius', FloatArg),
                                       ('height', FloatArg),
                                       ('from_point', CenterArg),
                                       ('to_point', CenterArg),
                                       ('axis', AxisArg),
                                       ('caps', BoolArg)] + common_args,
                            synopsis = 'create a cylinder model')
    register('shape cylinder', cylinder_desc, shape_cylinder, logger=logger)

    # Ellipsoid
    ellipsoid_desc = CmdDesc(keyword = [('radius', Float1or3Arg)] + common_args,
                             synopsis = 'create an ellipsoid model')
    register('shape ellipsoid', ellipsoid_desc, shape_sphere, logger=logger)

    # Icosahedron
    from chimerax.geometry.icosahedron import coordinate_system_names as icos_orientations
    icos_desc = CmdDesc(keyword = [('radius', FloatArg),
                                   ('sphere_factor', FloatArg),
                                   ('orientation', EnumOf(icos_orientations)),
                                   ('lattice', Int2Arg)] + common_args,
                          synopsis = 'create an icosahedron model')
    register('shape icosahedron', icos_desc, shape_icosahedron, logger=logger)

    # Rectangle
    rect_desc = CmdDesc(keyword = [('width', FloatArg),
                                   ('height', FloatArg),
                                   ('width_divisions', PositiveIntArg),
                                   ('height_divisions', PositiveIntArg)] + common_args,
                        synopsis = 'create a rectangle model')
    register('shape rectangle', rect_desc, shape_rectangle, logger=logger)

    # Ribbon
    ribbon_desc = CmdDesc(required = [('atoms', AtomsArg)],
                          keyword = [('width', FloatArg),
                                     ('height', FloatArg),
                                     ('yaxis', Float3Arg),
                                     ('twist', FloatArg),
                                     ('band_length', FloatArg),
                                     ('follow_bonds', BoolArg),
                                     ('segment_subdivisions', PositiveIntArg)] + basic_args,
                        synopsis = 'create a ribbon model')
    register('shape ribbon', ribbon_desc, shape_ribbon, logger=logger)

    # Sphere
    sphere_desc = CmdDesc(keyword = [('radius', Float1or3Arg)] + common_args,
                          synopsis = 'create a sphere model')
    register('shape sphere', sphere_desc, shape_sphere, logger=logger)
    
    # Triangle
    triangle_desc = CmdDesc(optional = [('atoms', AtomsArg)],
                            keyword = [('point', RepeatOf(CenterArg))] + common_args,
                            synopsis = 'create a triangle model')
    register('shape triangle', triangle_desc, shape_triangle, logger=logger)

    # Tube
    tube_desc = CmdDesc(required = [('atoms', AtomsArg)],
                        keyword = [('radius', FloatArg),
                                   ('band_length', FloatArg),
                                   ('follow_bonds', BoolArg),
                                   ('segment_subdivisions', PositiveIntArg)] + basic_args,
                        synopsis = 'create a tube model')
    register('shape tube', tube_desc, shape_tube, logger=logger)
