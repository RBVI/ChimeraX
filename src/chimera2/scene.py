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

from .math3d import Point, Vector, Xform, Identity, frustum, look_at, weighted_point, cross, BBox
from numpy import array, float32, uint8

bbox = BBox() #: The current bounding box.
_program_id = 0
_box_pn_id = None	# primitive box vertex position and normals
_box_pd = None
_box_indices_id = None	# primitive box indices
_box_indices = None

_glsl_version = '150'

def set_glsl_version(version):
	global _glsl_version
	if version not in ('150', 'webgl'):
		raise ValueError("Only support GLSL 150 and webgl (ES 1.0)")
	_glsl_version = version

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
	import os, sys
	shader_dir = os.path.dirname(__file__)
	_program_id = llgr.next_program_id()
	with open(os.path.join(shader_dir, "vertexShader_%s.txt" % _glsl_version)) as f:
		vertex_shader = f.read()
	with open(os.path.join(shader_dir, "fragmentShader_%s.txt" % _glsl_version)) as f:
		fragment_shader = f.read()
	with open(os.path.join(shader_dir, "vertexPickShader_%s.txt" % _glsl_version)) as f:
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
		rgba = array(color, dtype=float32)
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
		rgba = array(color, dtype=float32)
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
	#llb = Point(amin([p0, p1], axis=0))
	#urf = Point(amax([p0, p1], axis=0))
	#b = BBox(llb, urf)
	b = BBox()
	b.bulk_add([p0, p1])
	b.xform(xform)
	bbox.add_bbox(b)
	import llgr
	if isinstance(color, int):
		data_id = color
	else:
		data_id = llgr.next_data_id()
		assert len(color) == 4
		rgba = array(color, dtype=float32)
		llgr.create_singleton(data_id, rgba)

	if _box_pn_id is None:
		make_box_primitive()

	scale_id = llgr.next_data_id()
	scale = b.urf - b.llb
	llgr.create_singleton(scale_id, array(scale, dtype=float32))

	matrix_id = llgr.next_matrix_id()
	xform.translate(b.llb)
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

def render(viewport, vertical_fov, globalXform, as_string=False):
	"""render scene
	
	:param viewport: is a (lower-left, lower-right, width, height) tuple
	:param vertical_fov: is the veitical field of view in radians
	:param globalXform: is a :py:class:`~chimera2.math3d.Xform`
	   that rotates and translates the data after the camera is setup

	The camera is a simple one that takes the :param vertical_fov: and
	the current bounding box, and calculates the eye position and looks
	at the bounding box down the negative z-axis.

	There are two lights and the directions are fixed.
	"""
	import llgr
	from . import lighting
	zero = array([0, 0, 0, 0], dtype=float32)
	amb = lighting.ambient
	ambient = array([amb, amb, amb, 1], dtype=float32)
	if lighting.fill_light is None:
		f_diffuse = zero
		f_position = zero
	else:
		color = lighting.fill_light.color
		f_diffuse = array(color.rgb + [1], dtype=float32)
		direct = lighting.fill_light.direction
		f_position = array(list(direct) + [0], dtype=float32)
	if lighting.key_light is None:
		k_diffuse = zero
		k_specular = zero
		k_position = zero
	else:
		color = lighting.key_light.color
		k_diffuse = array(color.rgb + [1], dtype=float32)
		direct = lighting.key_light.direction
		k_position = array(list(direct) + [0], dtype=float32)
		reflectivity = lighting.reflectivity()
		color = lighting.shiny_color()
		specular = [x * reflectivity for x in color.rgb]
		k_specular = array(specular + [1], dtype=float32)
	shininess = array([lighting.sharpness()], dtype=float32)
	llgr.set_uniform(0, 'Ambient', llgr.FVec4, ambient)
	llgr.set_uniform(0, 'FillDiffuse', llgr.FVec4, f_diffuse)
	llgr.set_uniform(0, 'FillPosition', llgr.FVec4, f_position)
	llgr.set_uniform(0, 'KeyDiffuse', llgr.FVec4, k_diffuse)
	llgr.set_uniform(0, 'KeySpecular', llgr.FVec4, k_specular)
	llgr.set_uniform(0, 'KeyPosition', llgr.FVec4, k_position)
	llgr.set_uniform(0, 'Shininess', llgr.FVec1, shininess)

	llgr.set_clear_color(.05, .05, .4, 0)
	if not as_string:
		# TODO: move to llgr or to calling routine?
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

	if as_string:
		return llgr.render(as_string)
	else:
		llgr.render()
