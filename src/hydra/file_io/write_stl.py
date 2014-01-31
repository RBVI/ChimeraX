# -----------------------------------------------------------------------------
# Output Surface models in binary STL format.
#
def write_surfaces_as_stl(path, surfaces, session, displayed_only = True):

    if displayed_only:
        surfs = [s for s in surfaces if s.displayed]
        plist = sum(([p for p in s.surface_pieces() if p.display] for s in surfs), [])
    else:
        surfs = surfaces
        plist = sum((s.surface_pieces() for s in surfaces), [])
    f = open(path, 'wb')
    write_surface_pieces(plist, f)
    f.close()
    from . import fileicon
    fileicon.set_file_icon(path, session, models = surfs)

# -----------------------------------------------------------------------------
#
def write_stl_command(cmdname, args, session):

    from ..ui.commands import path_arg, surfaces_arg, bool_arg
    from ..ui.commands import parse_arguments
    req_args = (('path', path_arg),
                ('surfaces', surfaces_arg),
                )
    opt_args = ()
    kw_args = (('displayed_only', bool_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    write_surfaces_as_stl(**kw)

# -----------------------------------------------------------------------------
#
def write_surface_pieces(plist, file):

    # Write 80 character comment.
    from .. import version
    created_by = '# Created by Hydra %s' % version
    comment = created_by + ' ' * (80 - len(created_by))
    file.write(comment.encode('utf-8'))

    # Write number of triangles
    tc = 0
    for p in plist:
        varray,tarray = p.geometry
        tc += len(tarray)
    from numpy import uint32
    file.write(binary_bytes(tc, uint32))

    # Write triangles.
    # TODO: handle surface instances
    for p in plist:
        varray,tarray = p.geometry
        tf = p.surface.placement
        if not tf.is_identity():
            tf.move(varray)
        file.write(stl_triangle_geometry(varray, tarray))

# -----------------------------------------------------------------------------
#
def stl_triangle_geometry(varray, tarray):

    from numpy import empty, float32, little_endian
    ta = empty((12,), float32)

    slist = []
    abc = b'\0\0'
    for vi0,vi1,vi2 in tarray:
        v0,v1,v2 = varray[vi0],varray[vi1],varray[vi2]
        n = triangle_normal(v0,v1,v2)
        ta[:3] = n
        ta[3:6] = v0
        ta[6:9] = v1
        ta[9:12] = v2
        if not little_endian:
            ta[:] = ta.byteswap()
        slist.append(ta.tostring() + abc)
    g = b''.join(slist)
    return g

# -----------------------------------------------------------------------------
#
def triangle_normal(v0,v1,v2):

    e10, e20 = v1 - v0, v2 - v0
    from ..geometry import vector
    n = vector.normalize_vector(vector.cross_product(e10, e20))
    return n

# -----------------------------------------------------------------------------
#
def binary_bytes(x, dtype):

    from numpy import array, little_endian
    ta = array((x,), dtype)
    if not little_endian:
        ta[:] = ta.byteswap()
    return ta.tostring()
