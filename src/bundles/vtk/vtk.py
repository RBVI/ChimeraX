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

"""
vtk: VTK polydata file reader
=============================

Read ascii and binary VTK format files.
"""
def read_vtk(session, filename, name, *args, **kw):
    """
    Create a model showing point, lines, triangles, polygons.
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        fname = filename.name
        filename.close()
        f = open(fname, 'r')
    else:
        # TODO: will need binary mode for handling binary files.
        f = open(filename, 'r')
        fname = filename

    # parse file
    from chimerax.core.errors import UserError
    ver = f.readline()
    if not ver.startswith('# vtk DataFile Version'):
      raise UserError('First line does not start with "# vtk DataFile Version"')

    header = f.readline()               # description of data set
    
    ab = f.readline().strip()           # ASCII or BINARY
    if ab != 'ASCII':
      raise UserError('VTK file is not ascii format, got "%s"' % ab)

    g = f.readline().strip()
    if g != 'DATASET POLYDATA':
      raise UserError('VTK file is not structured points, got "%s"' % g)

    points = polylines = None
    while True:
        line = f.readline().strip()
        if not line:
            break
        if line.startswith('POINTS'):
            pf = line.split()
            if len(pf) != 3:
                raise UserError('VTK file POINTS line does not have 3 fields, got "%s"' % line)
            if pf[2] != 'float':
                raise UserError('VTK file POINTS are not float, got "%s"' % line)
            try:
                np = int(pf[1])
            except ValueError:
                raise UserError('VTK file POINTS count is not an integer, got "%s"' % line)
            points = read_ascii_floats(f, 3*np).reshape((np,3))
        elif line.startswith('LINES'):
            lf = line.split()
            if len(lf) != 3:
                raise UserError('VTK file LINES line does not have 3 fields, got "%s"' % line)
            try:
                nl,nlp = int(lf[1]), int(lf[2])
            except ValueError:
                raise UserError('VTK file LINES count not an integer, got "%s"' % line)
            polylines = []
            for i in range(nl):
                lif = f.readline().split()
                nlip = int(lif[0])
                polylines.append(tuple(int(lif[j+1]) for j in range(nlip)))
        else:
            raise UserError('VTK file line does not start with POINTS or LINES, got "%s"' % line)

    if points is None:
        raise UserError('VTK file did not contain POINTS')
    if polylines is None:
        raise UserError('VTK file did not contain LINES')

    from os.path import basename
    model = lines_model(session, points, polylines, name = basename(fname))
    
    return [model], ("Opened VTK file %s containing %d points, %d lines" % (fname, np, nl))

# -----------------------------------------------------------------------------
#
def read_ascii_floats(file, n):
    from numpy import empty, float32
    fv = empty((n,), float32)
    c = 0
    while c < n:
        line = file.readline()
        values = tuple(float(v) for v in line.split())
        nv = len(values)
        fv[c:c+nv] = values
        c += nv
    return fv

# -----------------------------------------------------------------------------
#
def lines_model(session, points, polylines, name = 'vtk lines', color = (255,255,255,255)):
    from chimerax.core.models import Model
    m = Model(name, session)
    m.vertices = points

    # Compute line segments array
    ns = sum(len(pl)-1 for pl in polylines)
    ls = []
    for pl in polylines:
        ls.extend(pl[i:i+2] for i in range(len(pl)-1))
        #ls.extend((pl[i],pl[i+1],pl[i+1]) for i in range(len(pl)-1))
    from numpy import array, int32
    m.triangles = array(ls, int32)
    m.display_style = m.Mesh
    m.color = color

    return m
    
# -----------------------------------------------------------------------------
#
def register():
    from chimerax.core import io, generic3d
    io.register_format(
        "VTK PolyData", generic3d.CATEGORY, (".vtk",), ("vtk",),
        reference="http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf",
        open_func=read_vtk)
