"""
scene: scene management
=======================

The scene module is placeholder for demostration purposes.
A real scene module would have camera, lights, and more.
Instead, the camera is computed from a field-of-view angle and viewport
when :py:func:`render` is called, the two lights have fixed directions,
and the names of the shader program uniforms are fixed too.

Geometry may added to the scene in two ways:
(1) by adding a known shape primitive, *e.g.*, with :py:func:`add_sphere`,
or (2) by creating geometry directly with :py:mod:`llgr`
and updating the scene's bounding box.
For example, given *xyzs* as an :py:class:`~numpy.array` of XYZ coordinates::

        import scene
        # if using a non-identity instance matrix, the coordinates would
        # have to be transformed first
        scene.bbox.bulk_add(xyzs)
	# llgr code
"""

__all__ = [
	'BBox', 'bbox', 
	'reset',
	'add_sphere',
	'add_cylinder',
	'add_box',
	'render'
]

from .math3d import Point, Vector, Xform, Identity, frustum, look_at, weighted_point, cross
from numpy import array, amin, amax, float32, uint8

class BBox:
	"""right-handed axis-aligned bounding box

	If either :py:attr:`BBox.llb` or :py:attr:`BBox.urf` are None,
	then the bounding box is uninitialized.
	"""

	__slots__ = ['llb', 'urf']

	def __init__(self, llb=None, urf=None):
		self.llb = None	#: lower-left-back corner coordinates, a :py:class:`~chimera2.math3d.Point`
		self.urf = None	#: upper-right-front corner coordinates, a :py:class:`~chimera2.math3d.Point`
		if llb is not None:
			self.llb = Point(llb)
		if urf is not None:
			self.urf = Point(urf)

	def add(self, pt):
		"""expand bounding box to encompass given point

		:param pt: a :py:class:`~chimera2.math3d.Point or other XYZ-tuple`
		"""
		if self.llb is None:
			self.llb = Point(pt)
			self.urf = Point(pt)
			return
		for i in range(3):
			if pt[i] < self.llb[i]:
				self.llb[i] = pt[i]
			elif pt[i] > self.urf[i]:
				self.urf[i] = pt[i]

	def add_bbox(self, box):
		"""expand bounding box to encompass given bounding box
		
		:param box: a :py:class:`BBox`
		"""
		if self.llb is None:
			self.llb = box.llb
			self.urf = box.urf
			return
		for i in range(3):
			if box.llb[i] < self.llb[i]:
				self.llb[i] = box.llb[i]
			if box.urf[i] > self.urf[i]:
				self.urf[i] = box.urf[i]

	def bulk_add(self, pts):
		"""expand bounding box to encompass all given points

		:param pts: a numpy array of XYZ coordinates
		"""
		mi = amin(pts, axis=0)
		ma = amax(pts, axis=0)
		if self.llb is None:
			self.llb = Point(mi)
			self.urf = Point(ma)
			return
		for i in range(3):
			if mi[i] < self.llb[i]:
				self.llb[i] = mi[i]
			if ma[i] > self.urf[i]:
				self.urf[i] = ma[i]

	def center(self):
		"""return center of bounding box
		
		:rtype: a :py:class:`~chimera2.math3d.Point`
		"""

		if self.llb is None:
			raise ValueError("empty bounding box")
		return weighted_point([self.llb, self.urf])

	def size(self):
		"""return length of sides of bounding box
		
		:rtype: a :py:class:`~chimera2.math3d.Vector`
		"""
		if self.llb is None:
			raise ValueError("empty bounding box")
		return self.urf - self.llb

	def xform(self, xf):
		"""transform bounding box in place"""
		if xf.isIdentity:
			return
		b = BBox([0., 0., 0.], [0., 0., 0.])
		for i in range(3):
			b.llb[i] = b.urf[i] = xf._matrix[i][3]
			for j in range(3):
				coeff = xf._matrix[i][j]
				if coeff == 0:
					continue
				if coeff > 0:
					b.llb[i] += self.llb[j] * coeff
					b.urf[i] += self.urf[j] * coeff
				else:
					b.llb[i] += self.urf[j] * coeff
					b.urf[i] += self.llb[j] * coeff
		self.llb = b.llb
		self.urf = b.urf

bbox = BBox() #: The current bounding box.
_program_id = 0
_box_pn_id = None	# primitive box vertex position and normals
_box_indices_id = None	# primitive box indices

def reset():
	"""reinitialze scene
	
	Removes all objects, resets lights, bounding box,
	viewing transformation.
	"""
	global bbox, _program_id, _box_pn_id, _box_indices_id
	_box_pn_id = None
	_box_indices_id = None
	import llgr
	llgr.clear_all()
	_program_id = llgr.next_program_id()
	with open("../shaders/vertexShader2.txt") as f:
		vertex_shader = f.read()
	with open("../shaders/fragmentShader2.txt") as f:
		fragment_shader = f.read()
	with open("../shaders/vertexPickShader.txt") as f:
		pick_vertex_shader = f.read()
	llgr.create_program(_program_id, vertex_shader, fragment_shader,
						pick_vertex_shader)
	bbox = BBox()

def add_sphere(radius, center, color, xform=None):
	"""add sphere to scene
	
	:param radius: the radius of the sphere
	:param center: the center of the sphere, :py:class:`~chimera2.math3d.Point`
	:param color: the RGBA color of the sphere (either a sequence of 4 floats, or an integer referring to a previously defined color)
	"""
	if xform is None:
		xform = Identity()
	else:
		xform = Xform(xform)
	import llgr
	b = BBox(center - radius, center + radius)
	b.xform(xform)
	bbox.add_bbox(b)
	if isinstance(color, int):
		data_id = color
	else:
		data_id = llgr.next_data_id()
		assert len(color) == 4
		rgba = array(color, dtype=float32).tostring()
		llgr.create_singleton(data_id, rgba)

	matrix_id = llgr.next_matrix_id()
	xform.translate(center)
	mat = xform.getWebGLMatrix()
	llgr.create_matrix(matrix_id, mat, False)

	obj_id = llgr.next_object_id()
	ai = llgr.AttributeInfo("color", data_id, 0, 0, 4, llgr.Float)
	llgr.add_sphere(obj_id, radius, _program_id, matrix_id, [ai])

def add_cylinder(radius, p0, p1, color, xform=None):
	"""add cylinder to scene
	
	:param radius: the radius of the cylinder
	:param p0: one endpoint of the cylinder, :py:class:`~chimera2.math3d.Point`
	:param p1: the other endpoint of the cylinder, :py:class:`~chimera2.math3d.Point`
	:param color: the RGBA color of the cylinder (either a sequence of 4 floats, or an integer referring to a previously defined color)
	"""
	if xform is None:
		xform = Identity()
	else:
		xform = Xform(xform)
	b = BBox(p0 - radius, p0 + radius)
	b.add(p1 - radius)
	b.add(p1 + radius)
	b.xform(xform)
	bbox.add_bbox(b)
	import llgr, math
	if isinstance(color, int):
		data_id = color
	else:
		data_id = llgr.next_data_id()
		assert len(color) == 4
		rgba = array(color, dtype=float32).tostring()
		llgr.create_singleton(data_id, rgba)

	# create translation matrix
	matrix_id = llgr.next_matrix_id()
	xform.translate(weighted_point([p0, p1]))
	delta = p1 - p0
	height = delta.length()
	cylAxis = Vector([0, 1, 0])
	axis = cross(cylAxis, delta)
	cosine = (cylAxis * delta) / height
	angle = math.acos(cosine)
	if axis.sqlength() > 0:
		xform.rotate(axis, angle)
	elif cosine < 0:	# delta == -cylAxis
		xform.rotate(1, 0, 0, 180)
	mat = xform.getWebGLMatrix()
	llgr.create_matrix(matrix_id, mat, False)

	obj_id = llgr.next_object_id()
	ai = llgr.AttributeInfo("color", data_id, 0, 0, 4, llgr.Float)
	llgr.add_cylinder(obj_id, radius, height, _program_id, matrix_id, [ai])

def make_box_primitive():
	global _box_pn_id, _box_indices_id
	global _box_pn, _box_indices
	import llgr

	#       v2 ---- v3
	#        |\      |\
	#        | v6 ---- v7 = urf
	#        |  |    | |
	#        |  |    | |
	# llb = v0 -|---v1 |
	#         \ |     \|
	#          v4 ---- v5
	#
	# WebGL does not support "flat" variables, so duplicate everything

	# interleave vertex position and normal (px, py, pz, nx, ny, nz)
	_box_pn = array([
		# -x, v0-v4-v2-v6
		[0, 0, 0,  -1, 0, 0],
		[0, 0, 1,  -1, 0, 0],
		[0, 1, 0,  -1, 0, 0],
		[0, 1, 1,  -1, 0, 0],

		# -y, v0-v1-v4-v5
		[0, 0, 0,  0, -1, 0],
		[1, 0, 0,  0, -1, 0],
		[0, 0, 1,  0, -1, 0],
		[1, 0, 1,  0, -1, 0],

		# -z, v1-v0-v3-v2
		[1, 0, 0,  0, 0, -1],
		[0, 0, 0,  0, 0, -1],
		[1, 1, 0,  0, 0, -1],
		[0, 1, 0,  0, 0, -1],

		# x, v5-v1-v7-v3
		[1, 0, 1,  1, 0, 0],
		[1, 0, 0,  1, 0, 0],
		[1, 1, 1,  1, 0, 0],
		[1, 1, 0,  1, 0, 0],

		# y, v3-v2-v7-v6
		[1, 1, 0,  0, 1, 0],
		[0, 1, 0,  0, 1, 0],
		[1, 1, 1,  0, 1, 0],
		[0, 1, 1,  0, 1, 0],

		# z, v4-v5-v6-v7
		[0, 0, 1,  0, 0, 1],
		[1, 0, 1,  0, 0, 1],
		[0, 1, 1,  0, 0, 1],
		[1, 1, 1,  0, 0, 1],
	], dtype=float32)
	_box_pn_id = llgr.next_data_id()
	llgr.create_buffer(_box_pn_id, llgr.ARRAY, _box_pn)

	_box_indices = array([
		[0, 1, 2], [2, 1, 3],		# -x
		[4, 5, 6], [6, 5, 7],		# -y
		[8, 9, 10], [10, 9, 11],	# -z
		[12, 13, 14], [14, 13, 15],	# x
		[16, 17, 18], [18, 17, 19],	# y
		[20, 21, 22], [22, 21, 23],	# z
	], dtype=uint8)
	_box_indices_id = llgr.next_data_id()
	llgr.create_buffer(_box_indices_id, llgr.ELEMENT_ARRAY, _box_indices)

def add_box(p0, p1, color, xform=None):
	if xform is None:
		xform = Identity()
	else:
		xform = Xform(xform)
	llb = Point(amin([p0, p1], axis=0))
	urf = Point(amax([p0, p1], axis=0))
	b = BBox(llb, urf)
	b.xform(xform)
	bbox.add_bbox(b)
	import llgr
	if isinstance(color, int):
		data_id = color
	else:
		data_id = llgr.next_data_id()
		assert len(color) == 4
		rgba = array(color, dtype=float32).tostring()
		llgr.create_singleton(data_id, rgba)

	if _box_pn_id is None:
		make_box_primitive()

	scale_id = llgr.next_data_id()
	scale = urf - llb
	llgr.create_singleton(scale_id, array(scale, dtype=float32))

	matrix_id = llgr.next_matrix_id()
	xform.translate(llb)
	mat = xform.getWebGLMatrix()
	llgr.create_matrix(matrix_id, mat, False)

	obj_id = llgr.next_object_id()
	AI = llgr.AttributeInfo
	ais = [
		AI("color", data_id, 0, 0, 4, llgr.Float),
		AI("position", _box_pn_id, 0, _box_pn[0].nbytes, 3, llgr.Float),
		AI("normal", _box_pn_id, 12, _box_pn[0].nbytes, 3, llgr.Float),
		AI("instanceScale", scale_id, 0, 0, 3, llgr.Float),
	]
	llgr.create_object(obj_id, _program_id, matrix_id, ais, llgr.Triangles,
		0, _box_indices.size, _box_indices_id, llgr.UByte)

def render(viewport, vertical_fov, globalXform):
	"""render scene
	
	:param viewport: is a (lower-left, lower-right, width, height) tuple
	:param vertical_fov: is the veitical field of view
	:param globalXform: is a :py:class:`~chimera2.math3d.Xform`
	   that rotates and translates the data after the camera is setup

	The camera is a simple one that takes the :param vertical_fov: and
	the current bounding box, and calculates the eye position and looks
	at the bounding box down the negative z-axis.

	There are two lights and the directions are fixed.
	"""
	import llgr
	one = array([1, 1, 1, 1], dtype=float32)
	ambient = array([0.197, 0.197, 0.197, 1], dtype=float32)
	diffuse0 = array([0.432, 0.432, 0.432, 1], dtype=float32)
	position0 = array([0.251, 0.251, 0.935, 0], dtype=float32)
	diffuse1 = array([0.746, 0.746, 0.746, 1], dtype=float32)
	position1 = array([-0.357, 0.66, 0.66, 0], dtype=float32)
	shininess = array([30], dtype=float32)
	llgr.set_uniform(0, 'Ambient', llgr.FVec4, ambient)
	llgr.set_uniform(0, 'FillDiffuse', llgr.FVec4, diffuse0)
	llgr.set_uniform(0, 'FillPosition', llgr.FVec4, position0)
	llgr.set_uniform(0, 'KeyDiffuse', llgr.FVec4, diffuse1)
	llgr.set_uniform(0, 'KeySpecular', llgr.FVec4, one)
	llgr.set_uniform(0, 'KeyPosition', llgr.FVec4, position1)
	llgr.set_uniform(0, 'Shininess', llgr.FVec1, shininess)

	llgr.set_clear_color(.5, .2, .2, 0)
	from OpenGL import GL
	#if self._samples >= 2:
	#	GL.glEnable(GL.GL_MULTISAMPLE)
	#GL.glEnable(GL.GL_CULL_FACE)
	GL.glViewport(*viewport)

	if bbox.llb is not None:
		import math
		# projection and modelview matrices
		win_aspect = float(viewport[2]) / viewport[3]
		width2, height2, depth2 = bbox.size() / 2 * 1.1	# extra 10%
		scene_aspect = width2 / height2
		center = bbox.center()
		if win_aspect > scene_aspect:
			width2 = height2 * win_aspect
		else:
			height2 = width2 / win_aspect

		near = height2 / math.tan(vertical_fov / 2)
		far = near + 2 * depth2
		eye = Point([center[0], center[1], center[2] + near + depth2])
		at = center
		up = Point([center[0], center[1] + 1, center[2]])

		#camera = perspective(vertical_fov, win_aspect, near, far)
		camera = frustum(-width2, width2, -height2, height2, near, far)
		projection = camera.getWebGLMatrix()
		mv = look_at(eye, at, up)
		modelview = mv * globalXform
		llgr.set_uniform_matrix(0, 'ProjectionMatrix', False,
						llgr.Mat4x4, projection)
		llgr.set_uniform_matrix(0, 'ModelViewMatrix', False,
				llgr.Mat4x4, modelview.getWebGLMatrix())
		llgr.set_uniform_matrix(0, 'NormalMatrix', False,
				llgr.Mat3x3, modelview.getWebGLRotationMatrix())

	llgr.render()
