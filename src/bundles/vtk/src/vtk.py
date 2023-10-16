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

"""
vtk: VTK polydata file reader
=============================

Read ascii and binary VTK format files.
"""
def read_vtk(session, filename, name):
    """
    Create a model showing point, lines, triangles, polygons.
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        fname = filename.name
        f = filename
    else:
        f = open(filename, 'rb')
        fname = filename

    # parse file
    from chimerax.core.errors import UserError
    ver = f.readline()
    if not ver.startswith(b'# vtk DataFile Version'):
      raise UserError('First line does not start with "# vtk DataFile Version"')

    header = f.readline()               # description of data set
    
    ab = f.readline().strip()           # ASCII or BINARY
    if ab != b'ASCII' and ab != b'BINARY':
      raise UserError('VTK file line 3 is not "ASCII" or "BINARY", got "%s"' % ab)
    binary = (ab == b'BINARY')

    g = f.readline().strip()
    if g != b'DATASET POLYDATA':
      raise UserError('VTK file is not structured points, got "%s"' % g)

    points = line_segments = triangles = None
    details = ''
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        data_type, nobj, nnum = parse_data_type_line(line)
        if data_type == 'POINTS':
            points = read_floats(f, 3*nobj, binary).reshape((nobj,3))
        elif data_type == 'LINES':
            plines = read_ints(f, nnum, binary)
            nseg = nnum-2*nobj
            line_segments = polyline_segments(plines, nseg)
            details += ', %d lines' % nobj
        elif data_type == 'POLYGONS':
            pgons = read_ints(f, nnum, binary)
            ntri = nnum-3*nobj
            triangles = polygon_triangles(pgons, ntri)
            details += ', %d polygons' % nobj
        elif data_type == 'POINT_DATA' or data_type == 'CELL_DATA': 
            # Don't handle point data such as surface normals and scalars or cell data (attributes of lines and polygons).
            break
        if binary:
            # There is a newline following binary data.
            nline = f.readline()
            if nline != b'\n':
                raise UserError('VTK file data after line "%s" does not end with newline, got "%s"'
                                % (line, nline))
            
    if points is None:
        raise UserError('VTK file did not contain POINTS')
    if line_segments is None and triangles is None:
        raise UserError('VTK file did not contain LINES nor POLYGONS')

    models = []
    from os.path import basename
    mname = basename(fname)
    if line_segments is not None:
        models.append(lines_model(session, points, line_segments, name = mname + ' lines'))
    if triangles is not None:
        models.append(triangles_model(session, points, triangles, name = mname + ' polygons'))

    msg = 'Opened VTK file %s containing %d points%s' % (fname, len(points), details)
    return models, msg

# -----------------------------------------------------------------------------
#
def parse_data_type_line(line):
    f = line.split()

    data_type = f[0].decode('utf-8') if f else None
    if data_type in ('POINT_DATA', 'CELL_DATA'):
        return data_type, 0, 0
    
    from chimerax.core.errors import UserError
    if len(f) != 3:
        raise UserError('VTK file data type line does not have 3 fields, got "%s"' % line)

    if data_type in ('VERTICES', 'TRIANGLE_STRIPS'):
        raise UserError('VTK reader does not handle type %s, got line "%s"' % (data_type, line))
    if not data_type in ('POINTS', 'LINES', 'POLYGONS'):
        raise UserError('VTK file line does not start with POINTS, LINES, or POLYGONS, got "%s"' % line)

    try:
        nobj = int(f[1])
    except ValueError:
        raise UserError('VTK file object count is not an integer, got "%s"' % line)

    if line.startswith(b'POINTS'):
        if f[2] != b'float':
            raise UserError('VTK file only handle float POINTS, got "%s"' % line)
        nnum = 3 * nobj
    else:
        try:
            nnum = int(f[2])
        except ValueError:
            raise UserError('VTK file value count not an integer, got "%s"' % line)

    return data_type, nobj, nnum

# -----------------------------------------------------------------------------
#
def read_floats(file, n, binary = False):
    from numpy import float32
    return read_values(file, n, float32, float, binary)

# -----------------------------------------------------------------------------
#
def read_ints(file, n, binary = False):
    from numpy import int32
    return read_values(file, n, int32, int, binary)

# -----------------------------------------------------------------------------
#
def read_values(file, n, numpy_dtype, py_type, binary = False):
    from numpy import empty, frombuffer, float32
    if binary:
        dtype = numpy_dtype()
        b = file.read(dtype.itemsize * n)
        fv = frombuffer(b, dtype)
        import sys
        if sys.byteorder == 'little':
            # VTK binary files are written in big-endian byte order.
            fv = fv.byteswap()
    else:
        fv = empty((n,), numpy_dtype)
        c = 0
        while c < n:
            line = file.readline()
            values = tuple(py_type(v) for v in line.split())
            nv = len(values)
            fv[c:c+nv] = values
            c += nv
    return fv

# -----------------------------------------------------------------------------
#
def polyline_segments(plines, nseg):
    from numpy import empty, int32
    seg = empty((nseg,2), int32)
    p = s = 0
    while s < nseg:
        np = plines[p]
        for j in range(p+1, p+np):
            seg[s,:] = (plines[j], plines[j+1])
            s += 1
        p += np+1
    return seg

# -----------------------------------------------------------------------------
#
def polygon_triangles(pgons, ntri):
    from numpy import empty, int32
    tri = empty((ntri,3), int32)
    p = t = 0
    while t < ntri:
        np = pgons[p]
        for j in range(p+2, p+np):
            tri[t,:] = (pgons[p+1], pgons[j], pgons[j+1])
            t += 1
        p += np+1
    return tri

# -----------------------------------------------------------------------------
#
def lines_model(session, points, line_segments, name = 'vtk lines', color = (255,255,255,255)):
    from chimerax.core.models import Model
    m = Model(name, session)
    m.set_geometry(points, None, line_segments)
    m.SESSION_SAVE_DRAWING = True  # Save lines in session
    m.display_style = m.Mesh
    m.color = color
    return m

# -----------------------------------------------------------------------------
#
def triangles_model(session, points, triangles,
                    name = 'vtk polygons', color = (180,180,180,255)):
    from chimerax.core.models import Surface
    m = Surface(name, session)
    m.SESSION_SAVE_DRAWING = True  # Save triangles in session
    from chimerax import surface
    normals = surface.calculate_vertex_normals(points, triangles)
    m.set_geometry(points, normals, triangles)
    m.color = color
    return m
