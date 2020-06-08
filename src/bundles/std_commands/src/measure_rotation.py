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
#
def measure_rotation(session, model, to_model,
                     show_axis = True, show_slabs = False,
                     color = (210, 210, 100, 255), color2 = (100, 149, 237, 255),
                     radius = None, length = None, width = None, thickness = None,
                     coordinate_system = None):

    tf = to_model.scene_position.inverse() * model.scene_position
    if coordinate_system:
        csys = coordinate_system.scene_position
        ctf = to_model.scene_position.inverse() * csys
        transform = ctf.inverse() * tf * ctf
        message = ('Position of %s relative to %s in %s coordinate system:\n%s'
                   % (model, to_model, coordinate_system, transform.description()))
    else:
        message = ('Position of %s relative to %s coordinates:\n%s'
                   % (model, to_model, tf.description()))
        transform = tf

    log = session.logger
    log.info(message)

    ra = tf.rotation_axis_and_angle()[1]
    log.status('Rotation angle %.2f degrees' % ra)

    if show_axis:
        _show_axis(session, tf, color, length, radius, to_model)

    if show_slabs:
        b = to_model.bounds()
        if b:
            _show_slabs(session, tf, color, color2, b.center(),
                        length, width, thickness, to_model)

    return transform

# -----------------------------------------------------------------------------
#
def _show_axis(session, tf, color, length, radius, coordinate_system):

    axis, axis_point, angle, axis_shift = tf.axis_center_angle_shift()
    if angle < 0.1:
        from chimerax.core.errors import UserError
        raise UserError('Rotation angle is near zero (%g degrees)' % angle)

    b = coordinate_system.bounds()
    if b is None:
        from chimerax.core.errors import UserError
        raise UserError('Model %s must be visible to show axis' % coordinate_system)

    from chimerax.geometry import project_to_axis
    axis_center = project_to_axis(b.center(), axis, axis_point)
    axis_length = b.width() if length is None else length
    hl = 0.5*axis_length
    ap1 = axis_center - hl*axis
    ap2 = axis_center + hl*axis

    from chimerax.markers import MarkerSet, create_link

    mset = MarkerSet(session, 'rotation axis')
    mset.scene_position = coordinate_system.scene_position

    r = 0.025 * axis_length if radius is None else radius
    m1 = mset.create_marker(ap1, color, r)
    m2 = mset.create_marker(ap2, color, r)
    b = create_link(m1, m2, color, r)
    b.halfbond = True

    session.models.add([mset])
    
    return mset

# -----------------------------------------------------------------------------
#
def _show_slabs(session, tf, color, to_color, center,
                length, width, thickness, coordinate_system):

    # Make schematic illustrating rotation
    sm = _transform_schematic(session, tf, center, color, to_color,
                              length, width, thickness)
    if sm:
        sm.name = 'rotation slabs'
        session.models.add([sm])
        sm.scene_position = coordinate_system.scene_position

    return sm
    
# -----------------------------------------------------------------------------
# Create a surface model showing two squares, the second being the transformed
# version of the first.  The first should pass through the center point.
# The two squares should have a common edge (rotation axis).
#
def _transform_schematic(session, transform, center, from_rgba, to_rgba,
                         length, width, thickness):

    varray, narray, tarray = _transform_square(transform, center, length, width, thickness)
    if varray is None:
        return None             # No rotation.

    from chimerax.core.models import Model, Surface

    s1 = Surface('slab 1', session)
    s1.set_geometry(varray, narray, tarray)
    s1.color = from_rgba

    s2 = Surface('slab 2', session)
    varray2 = transform * varray
    narray2 = transform.transform_vectors(narray)
    s2.set_geometry(varray2, narray2, tarray)
    s2.color = to_rgba
    
    m = Model('transform schematic', session)
    m.add([s1,s2])

    return m

# -----------------------------------------------------------------------------
#
def _transform_square(transform, center, length, width, thickness):

    axis, angle_deg = transform.rotation_axis_and_angle()
    trans = transform.translation()
    from chimerax.geometry import inner_product, cross_product, norm
    t1 = trans - axis * inner_product(trans, axis)
    t2 = cross_product(axis, t1)

    from math import pi, sin, cos
    angle = angle_deg * pi / 180
    sa, ca = sin(angle), cos(angle)
    if 1-ca == 0:
        return None     # No rotation
    axis_offset = t1*.5 + t2*(.5*sa/(1-ca))
    cd = center - axis_offset
    sq1 = cd - axis * inner_product(cd, axis)
    h1 = 2*sq1 if width is None else (width/norm(sq1)-1)*sq1
    sq2 = axis*norm(sq1)*2
    h2 = sq2 if length is None else (0.5*length/norm(sq2))*sq2 
    dz = cross_product(sq1, axis) * .05 # Thickness vector
    if thickness is not None:
        dz *= 0.5*thickness/norm(dz)
    c = center
    corners = [c - sq1 - h2 - dz, c + h1 - h2 - dz,
               c + h1 + h2 - dz, c - sq1 + h2 - dz,
               c - sq1 - h2 + dz, c + h1 - h2 + dz,
               c + h1 + h2 + dz, c - sq1 + h2 + dz]
    va,na,ta = _box_geometry(corners)
    return va,na,ta

# -----------------------------------------------------------------------------
#
def _box_geometry(corners):
    '''Use separate vertices and normals for each face so edges are sharp.'''
    from chimerax.geometry import normalize_vector
    nx = normalize_vector(corners[1] - corners[0])
    ny = normalize_vector(corners[3] - corners[0])
    nz = normalize_vector(corners[4] - corners[0])
    from numpy import concatenate, array, int32, float32
    varray = concatenate((corners, corners, corners)).astype(float32)
    narray = varray.copy()
    narray[:] = (-nz,-nz,-nz,-nz,nz,nz,nz,nz,
                 -ny,-ny,ny,ny,-ny,-ny,ny,ny,
                 -nx,nx,nx,-nx,-nx,nx,nx,-nx)
    bottom_top = [(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
    back_front = [(8+v0,8+v1,8+v2) for v0,v1,v2 in ((2,3,7),(2,7,6),(0,1,5),(0,5,4))]
    left_right = [(16+v0,16+v1,16+v2) for v0,v1,v2 in ((3,0,4),(3,4,7), (1,2,6),(1,6,5))]
    triangles = bottom_top + back_front + left_right
    tarray = array(triangles, int32)
    return varray, narray, tarray

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ModelArg, BoolArg, Color8Arg, FloatArg
    desc = CmdDesc(
        required = [('model', ModelArg)],
        keyword = [('to_model', ModelArg),
                   ('show_axis', BoolArg),
                   ('show_slabs', BoolArg),
                   ('color', Color8Arg),
                   ('color2', Color8Arg),
                   ('length', FloatArg),
                   ('radius', FloatArg),
                   ('width', FloatArg),
                   ('thickness', FloatArg),
                   ('coordinate_system', ModelArg)],
        required_arguments = ['to_model'],
        synopsis = 'measure rotation of one model relative to another')
    register('measure rotation', desc, measure_rotation, logger=logger)
