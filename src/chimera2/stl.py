"""
stl: STL format support
=======================

Read little-endian STL binary format.
"""

# code taken from chimera 1.7

from . import scene
from .cmds import UserError

_builtin_open = open

def open(filename, average_normals=True, *args, **kw):
	"""Populate the scene with the geometry from a STL file

	:param filename: either the name of a file or a file-like object
	
	Extra arguments are ignored.
	"""

	if hasattr(filename, 'read'):
		# it's really a file-like object
		input = filename
	else:
		input = _builtin_open(filename, 'rb')

	# parse input
	cur_color = [0.7, 0.7, 0.7, 1.0]

	# First read 80 byte comment line
	comment = input.read(80)

	# Next read uint32 triangle count.
	from numpy import fromstring, uint32, empty, float32, concatenate, array
	tc = fromstring(input.read(4), uint32)        # triangle count

	# Next read 50 bytes per triangle containing float32 normal vector
	# followed three float32 vertices, followed by two "attribute bytes"
	# sometimes used to hold color information, but ignored by this reader.
	nv = empty((tc, 12), float32)
	for t in range(tc):
		nt = input.read(12*4 + 2)
		nv[t, :] = fromstring(nt[:48], float32)

	if input != filename:
		input.close()

	va, na, ta = stl_geometry(nv, average_normals)

	scene.bbox.bulk_add(va)

	import llgr
	vn_id = llgr.next_data_id()
	vn = concatenate([va, na])
	llgr.create_buffer(vn_id, llgr.ARRAY, vn)
	tri_id = llgr.next_data_id()
	llgr.create_buffer(tri_id, llgr.ELEMENT_ARRAY, ta)
	color_id = llgr.next_data_id()
	rgba = array(cur_color, dtype=float32)
	llgr.create_singleton(color_id, rgba)
	uniform_scale_id = llgr.next_data_id()
	llgr.create_singleton(uniform_scale_id, array([1, 1, 1], dtype=float32))

	obj_id = llgr.next_object_id()
	AI = llgr.AttributeInfo
	mai = [
		AI("color", color_id, 0, 0, 4, llgr.Float),
		AI(scene.Position, vn_id, 0, 0, 3, llgr.Float),
		AI(scene.Normal, vn_id, va.nbytes, 0, 3, llgr.Float),
		AI("instanceScale", uniform_scale_id, 0, 0, 3, llgr.Float),
	]

	tc = len(ta)
	if tc >= pow(2, 16):
		index_type = llgr.UInt
	elif tc >= pow(2, 8):
		index_type = llgr.UShort
	else:
		index_type = llgr.UByte
	llgr.create_object(obj_id, scene._program_id, 0, mai, llgr.Triangles,
		0, ta.size, tri_id, index_type)


# -----------------------------------------------------------------------------
#
def stl_geometry(nv, average_normals=True):

    if not average_normals:
        return stl_geometry_with_creases(nv)

    tc = nv.shape[0]

    # Assign numbers to vertices.
    from numpy import empty, uint8, uint16, uint32, float32, zeros, sqrt, newaxis
    if tc >= pow(2, 16):
	index_type = uint32
    elif tc >= pow(2, 8):
	index_type = uint16
    else:
	index_type = uint8
    tri = empty((tc, 3), index_type)
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
  
# -----------------------------------------------------------------------------
#
def stl_geometry_with_creases(nv):

    # Combine identical vertices.  The must have identical normals too.
    from numpy import empty, uint32, float32
    tri = empty((tc, 3), uint32)
    vnum = {}
    for t in range(tc):
        normal = nv[t, 0:3]
        v0, v1, v2 = nv[t, 3:6], nv[t, 6:9], nv[t, 9:12]
        for a, v in enumerate((v0, v1, v2)):
            tri[t, a] = vnum.setdefault((v, normal), len(vnum))

    nv = len(vnum)
    vert = empty((vnum, 3), float32)
    normals = empty((vnum, 3), float32)
    for (v, n), vn in vnum.items():
        vert[vn, :] = v
        normals[vn, :] = n

  # If two triangle edges have the same vertex positions in opposite order
  # but use different normals then stictch them together with zero area
  # triangles.

  # TODO: Not finished.
