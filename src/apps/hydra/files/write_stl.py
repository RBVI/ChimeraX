# -----------------------------------------------------------------------------
# Output Surface models in binary STL format.
#
def write_surfaces_as_stl(path, surfaces, session, displayed_only = True):

    f = open(path, 'wb')
    write_drawings(surfaces, f, displayed_only)
    f.close()

    from . import fileicon
    fileicon.set_file_icon(path, session, models = surfaces)

    session.file_history.add_entry(path, models = surfaces)

# -----------------------------------------------------------------------------
#
def write_stl_command(cmdname, args, session):

    from ..commands.parse import path_arg, models_arg, bool_arg, parse_arguments
    req_args = (('path', path_arg),
                ('surfaces', models_arg),
                )
    opt_args = ()
    kw_args = (('displayed_only', bool_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    write_surfaces_as_stl(**kw)

# -----------------------------------------------------------------------------
#
def write_drawings(surfaces, file, displayed_only):

    # Write 80 character comment.
    from .. import version
    created_by = '# Created by Hydra %s' % version
    comment = created_by + ' ' * (80 - len(created_by))
    file.write(comment.encode('utf-8'))

    # Write number of triangles
    tc = sum(s.number_of_triangles(displayed_only) for s in surfaces)
    from numpy import uint32
    file.write(binary_bytes(tc, uint32))

    # Write triangles.
    for s in surfaces:
        write_drawing(s, file, displayed_only)

# -----------------------------------------------------------------------------
#
def write_drawing(surf, file, displayed_only, place = None):

    varray,tarray = surf.geometry
    for p in surf.get_positions(displayed_only):
        pl = place*p if place else p
        if not varray is None:
            file.write(stl_triangle_geometry(pl.moved(varray), tarray))
        for d in surf.child_drawings():
            write_drawing(d, file, displayed_only, pl)

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
