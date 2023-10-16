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
stl: STL format support
=======================

Read and write STL files in binary (little-endian) or ascii format.
"""

# If STL_STATE_VERSION changes, then bump the bundle's
# (maximum) session version number.
STL_STATE_VERSION = 1

from chimerax.core.models import Surface
class STLModel(Surface):
    clip_cap = True
    SESSION_SAVE_DRAWING = True

    @property
    def num_triangles(self):
        """Return number of triangles in model."""
        return len(self.triangles)

    def triangle_info(self, n):
        """Return information about triangle ``n``."""
        return TriangleInfo(self, n)

from chimerax.core.state import State
class TriangleInfo(State):
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

    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        return {'stl model': self._stl, 'triangle index': self._index, 'version':STL_STATE_VERSION}

    @staticmethod
    def restore_snapshot(session, data):
        return TriangleInfo(data['stl model'], data['triangle index'])

# -----------------------------------------------------------------------------
#
def read_stl(session, path, name):
    """Populate the scene with the geometry from a STL file

    Parameters
    ----------
    path : string
       Path to file
    name : string
       Name of model that will be returned.
    """

    if stl_is_ascii(path):
        va, na, ta = read_ascii_stl_geometry(path)
    else:
        va, na, ta = read_binary_stl_geometry(path)

    model = STLModel(name, session)
    model.set_geometry(va, na, ta)
    cur_color = [0.7, 0.7, 0.7, 1.0]
    from numpy import array, uint8
    cur_color = (array(cur_color) * 255).astype(uint8)
    model.color = cur_color

    from os.path import basename
    msg = ("Opened STL file %s containing %d triangles"
           % (basename(path), len(model.triangles)))
    
    return [model], msg

# -----------------------------------------------------------------------------
#
def stl_is_ascii(path):
    '''ASCII STL files start with "solid".'''
    f = open(path, 'rb')
    start = f.read(5)
    f.close()
    return start == b'solid'

# -----------------------------------------------------------------------------
#
def read_binary_stl_geometry(path, replace_zero_normals = True):
    input = open(path, 'rb')
    comment = input.read(80)
    
    # Next read uint32 triangle count.
    from numpy import fromstring, uint32, float32, array, uint8
    tc = fromstring(input.read(4), uint32)[0]        # triangle count

    geom = input.read(tc*50)	# 12 floats per triangle, plus 2 bytes padding.
    input.close()
    
    if len(geom) < tc*50:
        from chimerax.core.errors import UserError
        raise UserError('STL file is truncated.  Header says it contains %d triangles, but only %d were in file.'
                        % (tc, len(geom) // 50))

    from .stl_cpp import stl_unpack
    va, na, ta = stl_unpack(geom)    # vertices, normals, triangles

    if replace_zero_normals and len(na) > 0 and (na[0] == (0,0,0)).all():
        from chimerax.surface import calculate_vertex_normals
        na = calculate_vertex_normals(va, ta)

    return va, na, ta

# -----------------------------------------------------------------------------
#
def read_ascii_stl_geometry(path):
    '''
solid name
facet normal ni nj nk
    outer loop
        vertex v1x v1y v1z
        vertex v2x v2y v2z
        vertex v3x v3y v3z
    endloop
endfacet
endsolid name
    '''

    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    vlist = []
    nlist = []

    for i in range(1,len(lines)):
        line = lines[i].strip()
        if line == '':
            continue
        
        if line.startswith('facet normal '):
            n = tuple(float(x) for x in line.split()[2:5])
        elif line.startswith('vertex '):
            v = tuple(float(x) for x in line.split()[1:4])
            vlist.append(v)
            nlist.append(n)
        elif line in ('outer loop', 'endloop', 'endfacet') or line.startswith('endsolid'):
            pass
        else:
            from chimerax.core.errors import UserError
            raise UserError('STL file line %d, bad format "%s"' % (i+1, line))

    nlist = replace_zero_normals(nlist, vlist)
    va, na, ta = merge_triangle_vertices(vlist, nlist)
    
    return va, na, ta

def replace_zero_normals(nlist, vlist):
    if have_zero_normals(nlist):
        na = triangle_normals(vlist)
        zero = (0,0,0)
        nlist = [(tuple(n2) if n1 == zero else n1) for n1,n2 in zip(nlist, na)]
    return nlist

def have_zero_normals(nlist):
    zero = (0,0,0)
    for n in nlist:
        if n == zero:
            return True
    return False

def triangle_normals(vlist):
    nt = len(vlist)//3
    from numpy import array, int32
    ta = array(range(3*nt), int32).reshape((nt,3))
    from chimerax.surface import calculate_vertex_normals
    na = calculate_vertex_normals(vlist, ta)
    return na

def merge_triangle_vertices(vlist, nlist):
    '''
    Combine duplicate vertices.
    '''
    uvlist = []
    vnlist = []
    tlist = []
    vindex = {}		# Map vertex float 3-tuple to unique index
    for v,n in zip(vlist, nlist):
        if v in vindex:
            vi = vindex[v]
            vnlist[vi].append(n)
        else:
            vindex[v] = vi = len(vindex)
            uvlist.append(v)
            vnlist.append([n])
        tlist.append(vi)

    # Use average normal at each vertex from all joining triangles.
    from numpy import array, float32, int32
    unlist = [array(vn, float32).mean(axis=0) for vn in vnlist]

    # Convert to numpy arrays
    va = array(uvlist, float32)
    na = array(unlist, float32)
    from chimerax.geometry import normalize_vectors
    normalize_vectors(na)
    nt = len(tlist)//3
    ta = array(tlist, int32).reshape((nt,3))
    
    return va, na, ta
        
# -----------------------------------------------------------------------------
#
def stl_unpack(geom):

    tc = len(geom) // 50
    
    # Next read 50 bytes per triangle containing float32 normal vector
    # followed three float32 vertices, followed by two "attribute bytes"
    # sometimes used to hold color information, but ignored by this reader.
    from numpy import empty, float32, fromstring
    nv = empty((tc, 12), float32)
    for t in range(tc):
        nv[t, :] = fromstring(geom[50*t:50*t+48], float32)

    # Assign numbers to vertices.
    from numpy import empty, int32, float32, zeros, sqrt, newaxis
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
    normals /= sqrt((normals * normals).sum(1))[:,newaxis]

    return vert, normals, tri

# -----------------------------------------------------------------------------
#
def write_stl(session, filename, models):
    if models is None:
        models = session.models.list()

    # Collect all drawing children of models.
    drawings = set()
    for m in models:
        if not m in drawings:
            for d in m.all_drawings():
                drawings.add(d)
            
    # Collect geometry, not including children, handle instancing
    geom = []
    for d in drawings:
        va, ta = d.vertices, d.masked_triangles
        if va is not None and ta is not None and d.display and d.parents_displayed:
            pos = d.get_scene_positions(displayed_only = True)
            if len(pos) > 0:
                geom.append((va, ta, pos))
    from chimerax.surface import combine_geometry_vtp
    va, ta = combine_geometry_vtp(geom)
    from .stl_cpp import stl_pack
    stl_geom = stl_pack(va, ta)
    
    # Write 80 character comment.
    from chimerax import app_dirs as ad
    version  = "%s %s version: %s" % (ad.appauthor, ad.appname, ad.version)
    created_by = '# Created by %s' % version
    comment = created_by + ' ' * (80 - len(created_by))

    file = open(filename, 'wb')
    file.write(comment.encode('utf-8'))

    # Write number of triangles
    tc = len(ta)
    from numpy import uint32
    file.write(binary_string(tc, uint32))

    # Write triangles.
    file.write(stl_geom)
    file.close()

# -----------------------------------------------------------------------------
#
def stl_pack(varray, tarray):

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
        slist.append(ta.tobytes() + abc)
    g = b''.join(slist)
    return g

# -----------------------------------------------------------------------------
#
def triangle_normal(v0,v1,v2):

    e10, e20 = v1 - v0, v2 - v0
    from chimerax.geometry import normalize_vector, cross_product
    n = normalize_vector(cross_product(e10, e20))
    return n

# -----------------------------------------------------------------------------
#
def binary_string(x, dtype):

    from numpy import array, little_endian
    ta = array((x,), dtype)
    if not little_endian:
        ta[:] = ta.byteswap()
    return ta.tobytes()
