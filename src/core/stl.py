# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
stl: STL format support
=======================

Read little-endian STL binary format.
"""

# code taken from chimera 1.7

_builtin_open = open
from . import generic3d


class STLModel(generic3d.Generic3DModel):
    @property
    def num_triangles(self):
        """Return number of triangles in model."""
        return len(self.triangles)

    def triangle_info(self, n):
        """Return information about triangle ``n``."""
        return TriangleInfo(self, n)


class TriangleInfo:
    """Information about an STL triangle."""
    def __init__(self, stl, index):
        self._stl = stl
        self._index = index

    def model(self):
        """Return STL model containing triangle."""
        return self._stl

    def index(self):
        """Return index of triangle in STL model."""
        return self._index

    def color(self):
        """Return color of triangle."""
        return self._stl.color

    def coords(self):
        """Return coordinates of each vertex of triangles."""
        return self._stl.vertices[self._stl.triangles[self._index]]


def open(session, filename, name, *args, **kw):
    """Populate the scene with the geometry from a STL file

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        input = filename
    else:
        input = _builtin_open(filename, 'rb')

    model = STLModel(name)

    # parse input:

    # First read 80 byte comment line
    comment = input.read(80)
    del comment

    # Next read uint32 triangle count.
    from numpy import fromstring, uint32, empty, float32, array, uint8
    tc = fromstring(input.read(4), uint32)        # triangle count

    # Next read 50 bytes per triangle containing float32 normal vector
    # followed three float32 vertices, followed by two "attribute bytes"
    # sometimes used to hold color information, but ignored by this reader.
    nv = empty((tc, 12), float32)
    for t in range(tc):
        nt = input.read(12 * 4 + 2)
        nv[t, :] = fromstring(nt[:48], float32)

    if input != filename:
        input.close()

    va, na, ta = stl_geometry(nv)    # vertices, normals, triangles
    model.vertices = va
    model.normals = na
    model.triangles = ta
    cur_color = [0.7, 0.7, 0.7, 1.0]
    cur_color = (array(cur_color) * 255).astype(uint8)
    model.color = cur_color
    return [model], ("Opened STL file containing %d triangles"
                     % len(model.triangles))


def stl_geometry(nv):
    tc = nv.shape[0]

    # Assign numbers to vertices.
    from numpy import (
        empty, int32, float32, zeros, sqrt, newaxis
    )
    tri = empty((tc, 3), int32)
    vnum = {}
    for t in range(tc):
        v0, v1, v2 = nv[t, 3:6], nv[t, 6:9], nv[t, 9:12]
        for a, v in enumerate((v0, v1, v2)):
            tri[t, a] = vnum.setdefault(tuple(v), len(vnum))

    # Make vertex coordinate array.
    vc = len(vnum)
    vert = empty((vc, 3), float32)
    for v, vn in vnum.items():
        vert[vn, :] = v

    # Make average normals array.
    normals = zeros((vc, 3), float32)
    for t, tvi in enumerate(tri):
        for i in tvi:
            normals[i, :] += nv[t, 0:3]
    normals /= sqrt((normals ** 2).sum(1))[:, newaxis]

    return vert, normals, tri


def register():
    from . import io
    io.register_format(
        "StereoLithography", generic3d.CATEGORY, (".stl",),
        reference="http://en.wikipedia.org/wiki/STL_%28file_format%29",
        open_func=open)
