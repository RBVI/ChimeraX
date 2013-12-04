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
	'set_glsl_version',
	'Camera',
	'Graphics',
	'View',
]

from .math3d import (Point, weighted_point, Vector, cross,
		Xform, Identity, Rotation, Translation,
		frustum, ortho, look_at, camera_orientation, BBox)
from numpy import array, float32, uint, uint16, uint8, concatenate
from math import radians
from .trackchanges import track
from collections import Counter
from .lighting import lighting, Lighting

@track.register_data_type
class Camera:

	# modified reasons
	CAMERA_RESET = 'camera reset'
	VIEWPORT_CHANGE = 'viewport changed'
	CAMERA_MOVED = 'camera moved'

	def __init__(self, width, height, fov, bbox):
		self.eye = Point([0, 0, 0])
		self.at = Point([0, 0, -1])
		self.up = Vector([0, 1, 0])
		self.ortho = False
		self.reset(width, height, fov, bbox, _track=False)
		track.created(Camera, [self])

	def reset(self, width, height, fov, bbox, _track=True):
		# The camera is a simple one that takes the :param fov: and
		# the current bounding box, and calculates the eye position
		# and looks at the bounding box down the negative z-axis.
		import math, copy
		# projection and modelview matrices
		self.bbox = copy.deepcopy(bbox)
		self.at = self.bbox.center()
		self.width2, self.height2, depth2 = self.bbox.size() / 2 * 1.1	# +10%
		self.update_viewport(width, height, _track)

		self.near = self.height2 / math.tan(fov / 2)
		self.far = self.near + 2 * depth2
		self.eye = self.at + Vector([0, 0, self.near + depth2])
		self.up = Vector([0, 1, 0])
		if _track:
			track.modified(Camera, [self], self.CAMERA_RESET)

	def update_viewport(self, width, height, _track=True):
		win_aspect = width / height
		scene_aspect = self.width2 / self.height2
		if win_aspect > scene_aspect:
			self.width2 = self.height2 * win_aspect
		else:
			self.height2 = self.width2 / win_aspect
		if _track:
			track.modified(Camera, [self], self.VIEWPORT_CHANGE)

	def matrices(self):
		if self.ortho:
			projection = ortho(-self.width2, self.width2,
				-self.height2, self.height2,
				self.near, self.far)
		else:
			projection = frustum(-self.width2, self.width2,
				-self.height2, self.height2,
				self.near, self.far)
		modelview = look_at(self.eye, self.at, self.up)
		return projection, modelview

	def rotate(self, axis, angle):
		xf = Rotation(axis, angle)
		self.xform(xf)

	def xform(self, xf):
		if not xf._pure:
			raise ValueError('only pure rotation is allowed')
		modelview = camera_orientation(self.eye, self.at, self.up)
		inv_modelview = modelview.inverse()
		nxf = inv_modelview * xf.inverse() * modelview
		self.eye = nxf * self.eye
		self.up = nxf * self.up
		self.at = nxf * self.at
		track.modified(Camera, [self], self.CAMERA_MOVED)

_program_id = 0
_box_pn_id = None	# primitive box vertex position and normals
_box_pn = None
_box_indices_id = None	# primitive box indices
_box_indices = None

_glsl_version = '150'

def set_glsl_version(version):
	global _glsl_version
	if version not in ('150', 'webgl'):
		raise ValueError("Only support GLSL 150 and webgl (ES 1.0)")
	_glsl_version = version

@track.register_data_type(after=[Camera, Lighting])
class View:

	# modified reasons
	OPEN_MODELS_CHANGE = 'open models update'
	CAMERA_CHANGE = 'camera update'
	GRAPHICS_CHANGE = 'graphics update'
	FOV_CHANGE = 'fov changed'
	VIEWPORT_CHANGE = 'viewport changed'
	LIGHTING_CHANGE = lighting.LIGHTING_CHANGE

	def __init__(self, models=None):
		# 'models is None' means to track open models
		# TODO: support other values of models that allow for
		#       different set of models (e.g., individual model views,
		#       lighting UI, etc.)
		self._bbox = BBox()		 #: The current bounding box.
		self._camera = None
		self._fov = radians(30)
		self._viewport = (200, 200)
		self._omh = None		# OpenModels handler
		self._gh = None			# Graphics handler
		self._ch = None			# Camera handler
		if models is not None:
			self._models = models
		else:
			from .trackchanges import track
			from .open_models import OpenModels
			self._omh = track.add_handler(OpenModels,
						self._update_open_models)
			from . import open_models
			self._models = open_models.list()
		self._num_models = len(self._models)
		if self._models:
			self._num_models = len(models)
			self.reset_camera()
		self._gh = track.add_handler(Graphics, self._update_graphics)
		self._lh = track.add_handler(Lighting, self._update_lighting)
		track.created(View, [self])

	def _update_open_models(self, ignore_open_models):
		from . import open_models
		models = open_models.list()
		if self._models == models:
			return
		old_num = self._num_models
		self._models = models
		self._num_models = len(models)
		# want transitions from 0 to 1 and vice-versa
		if (old_num == 0 and self._num_models == 1) \
		or (old_num > 0 and self._num_models == 0):
			self.reset_camera()
		track.modified(View, [self], self.OPEN_MODELS_CHANGE)

	def reset_camera(self):
		bbox = BBox()
		for m in self._models:
			if m.graphics:
				bbox.merge(m.graphics.bbox)
		self._bbox = bbox
		if self._ch is not None:
			track.delete_handler(self._ch)
			self._ch = None
		if bbox.llb is None:
			self._camera = None
		else:
			self._camera = Camera(self._viewport[0], self._viewport[1], self._fov, bbox)
			self._ch = track.add_handler(Camera, self._update_camera)

	def _update_camera(self, cameras):
		if self.camera not in cameras.modified and self.camera not in cameras.created:
			return
		track.modified(View, [self], self.CAMERA_CHANGE)

	def _update_graphics(self, graphics):
		if not graphics.modified:
			return
		my_graphics = set([m.graphics for m in self._models if m.graphics is not None])
		if not graphics.modified.issubset(my_graphics):
			return
		track.modified(View, [self], self.GRAPHICS_CHANGE)

	def _update_lighting(self, lighting):
		track.modified(View, [self], self.LIGHTING_CHANGE)

	@property
	def models(self):
		return self._models

	@property
	def bbox(self):
		return self._bbox

	@property
	def camera(self):
		return self._camera

	@property
	def fov(self):
		return self._fov

	@fov.setter
	def fov(self, fov):
		# :param fov: is the vertical field of view in radians
		if fov == self._fov:
			return
		self._fov = fov
		track.modified(View, [self], self.FOV_CHANGE)

	@property
	def viewport(self):
		return self._viewport

	@viewport.setter
	def viewport(self, wh):
		# :param viewport: is a (lower-left, lower-right, width, height) tuple
		if self._viewport == wh:
			return
		self._viewport = wh
		if self.camera:
			width, height = wh
			self.camera.update_viewport(width, height)
		track.modified(View, [self], self.VIEWPORT_CHANGE)

	def reset(self):
		"""reinitialze view
		
		Removes all objects, resets lights, bounding box,
		viewing transformation.
		"""
		for m in self._models:
			if m.graphics is not None:
				m.graphics.clear()
		self._bbox = BBox()
		self._camera = None

	def render(self, as_data=False, skip_camera_matrices=False):
		"""render view
		"""
		import llgr
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
			reflectivity = lighting.reflectivity
			color = lighting.shiny_color
			specular = [x * reflectivity for x in color.rgb]
			k_specular = array(specular + [1], dtype=float32)
		shininess = array([lighting.sharpness], dtype=float32)
		llgr.set_uniform(0, 'Ambient', llgr.FVec4, ambient)
		llgr.set_uniform(0, 'FillDiffuse', llgr.FVec4, f_diffuse)
		llgr.set_uniform(0, 'FillPosition', llgr.FVec4, f_position)
		llgr.set_uniform(0, 'KeyDiffuse', llgr.FVec4, k_diffuse)
		llgr.set_uniform(0, 'KeySpecular', llgr.FVec4, k_specular)
		llgr.set_uniform(0, 'KeyPosition', llgr.FVec4, k_position)
		llgr.set_uniform(0, 'Shininess', llgr.FVec1, shininess)

		llgr.set_clear_color(.05, .05, .4, 0)
		if llgr.output_type().endswith('opengl'):
			# TODO: move to llgr or to calling routine?
			from OpenGL import GL
			#if self._samples >= 2:
			#	GL.glEnable(GL.GL_MULTISAMPLE)
			#GL.glEnable(GL.GL_CULL_FACE)
			GL.glViewport(0, 0, self._viewport[0], self._viewport[1])

		if self._camera and not skip_camera_matrices:
			projection, modelview = self._camera.matrices()
			llgr.set_uniform_matrix(0, 'ProjectionMatrix', False,
				llgr.Mat4x4, projection.getWebGLMatrix())
			llgr.set_uniform_matrix(0, 'ModelViewMatrix', False,
				llgr.Mat4x4, modelview.getWebGLMatrix())
			llgr.set_uniform_matrix(0, 'NormalMatrix', False,
				llgr.Mat3x3, modelview.getWebGLRotationMatrix())

		if as_data:
			return llgr.render(as_data)
		else:
			llgr.render()

def reset():
	"""reinitialze scene
	
	Removes all objects, resets lights, bounding box,
	viewing transformation.
	"""
	import llgr
	global _program_id, _box_pn_id, _box_indices_id
	_box_pn_id = None
	_box_indices_id = None
	llgr.clear_all()
	import os
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
#TODO	for each view:
#		view.make_graphics()

@track.register_data_type(before=[View])
class Graphics:

	# modified reasons
	MORE_OBJECTS = 'more objects'
	LESS_OBJECTS = 'less objects'

	def __init__(self):
		self.bbox = BBox()
		self.object_ids = set()
		self.matrix_ids = Counter()
		self.data_ids = Counter()
		self.group_id = None
		track.created(Graphics, [self])

	def _new_group(self):
		import llgr
		self.group_id = llgr.next_group_id()
		llgr.create_group(self.group_id, [])

	def add_sphere(self, radius, center, color, xform=None):
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
		self.bbox.add_bbox(b)
		if isinstance(color, int):
			color_id = color
		else:
			color_id = llgr.next_data_id()
			assert len(color) == 4
			rgba = array(color, dtype=float32)
			llgr.create_singleton(color_id, rgba)

		matrix_id = llgr.next_matrix_id()
		xform.translate(center)
		mat = xform.getWebGLMatrix()
		llgr.create_matrix(matrix_id, mat, False)

		obj_id = llgr.next_object_id()
		ai = llgr.AttributeInfo("color", color_id, 0, 0, 4, llgr.Float)
		llgr.add_sphere(obj_id, radius, _program_id, matrix_id, [ai])
		if self.group_id is None:
			self._new_group()
		llgr.group_add(self.group_id, obj_id)
		self.data_ids.update([color_id])
		self.matrix_ids.update([matrix_id])
		self.object_ids.update([obj_id])
		track.modified(Graphics, [self], self.MORE_OBJECTS)

	def add_cylinder(self, radius, p0, p1, color, xform=None):
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
		self.bbox.add_bbox(b)
		import llgr, math
		if isinstance(color, int):
			color_id = color
		else:
			color_id = llgr.next_data_id()
			assert len(color) == 4
			rgba = array(color, dtype=float32)
			llgr.create_singleton(color_id, rgba)

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
		ai = llgr.AttributeInfo("color", color_id, 0, 0, 4, llgr.Float)
		llgr.add_cylinder(obj_id, radius, height, _program_id, matrix_id, [ai])
		if self.group_id is None:
			self._new_group()
		llgr.group_add(self.group_id, obj_id)
		self.data_ids.update([color_id])
		self.matrix_ids.update([matrix_id])
		self.object_ids.update([obj_id])
		track.modified(Graphics, [self], self.MORE_OBJECTS)

	def add_box(self, p0, p1, color, xform=None):
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
		self.bbox.add_bbox(b)
		import llgr
		if isinstance(color, int):
			color_id = color
		else:
			color_id = llgr.next_data_id()
			assert len(color) == 4
			rgba = array(color, dtype=float32)
			llgr.create_singleton(color_id, rgba)

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
			AI("color", color_id, 0, 0, 4, llgr.Float),
			AI("position", _box_pn_id, 0, _box_pn[0].nbytes, 3, llgr.Float),
			AI("normal", _box_pn_id, 12, _box_pn[0].nbytes, 3, llgr.Float),
			AI("instanceScale", scale_id, 0, 0, 3, llgr.Float),
		]
		llgr.create_object(obj_id, _program_id, matrix_id, ais, llgr.Triangles,
			0, _box_indices.size, _box_indices_id, llgr.UByte)
		if self.group_id is None:
			self._new_group()
		llgr.group_add(self.group_id, obj_id)
		self.data_ids.update([color_id, scale_id])
		self.matrix_ids.update([matrix_id])
		self.object_ids.update([obj_id])
		track.modified(Graphics, [self], self.MORE_OBJECTS)

	def add_triangles(self, vertices, normals, color, indices):
		# vertices: Nx3 numpy array of float32 vertices
		# normals: Nx3 numpy array of float32 normals
		# color: a color_id or a length 4 collection of RGBA
		# indices: Nx3 numpy array of indices

		self.bbox.bulk_add(vertices)

		import llgr
		vn_id = llgr.next_data_id()
		vn = concatenate([vertices, normals])
		llgr.create_buffer(vn_id, llgr.ARRAY, vn)
		tc = len(vertices)
		if tc >= pow(2, 16):
			index_type = llgr.UInt
			ta = array(indices, dtype=uint)
		elif tc >= pow(2, 8):
			index_type = llgr.UShort
			ta = array(indices, dtype=uint16)
		else:
			index_type = llgr.UByte
			ta = array(indices, dtype=uint8)
		tri_id = llgr.next_data_id()
		llgr.create_buffer(tri_id, llgr.ELEMENT_ARRAY, ta)
		if isinstance(color, int):
			color_id = color
		else:
			color_id = llgr.next_data_id()
			assert len(color) == 4
			rgba = array(color, dtype=float32)
			llgr.create_singleton(color_id, rgba)
		scale_id = llgr.next_data_id()
		llgr.create_singleton(scale_id, array([1, 1, 1], dtype=float32))

		matrix_id = 0		# default identity matrix

		obj_id = llgr.next_object_id()
		AI = llgr.AttributeInfo
		ais = [
			AI("color", color_id, 0, 0, 4, llgr.Float),
			AI("position", vn_id, 0, 0, 3, llgr.Float),
			AI("normal", vn_id, vertices.nbytes, 0, 3, llgr.Float),
			AI("instanceScale", scale_id, 0, 0, 3, llgr.Float),
		]

		llgr.create_object(obj_id, _program_id, matrix_id, ais,
				llgr.Triangles, 0, ta.size, tri_id, index_type)
		self.data_ids.update([vn_id, tri_id, color_id, scale_id])
		if matrix_id:
			self.matrix_ids.update([matrix_id])
		self.object_ids.update([obj_id])
		track.modified(Graphics, [self], self.MORE_OBJECTS)

	def clear(self):
		import llgr
		self.data_ids.subtract([_box_pn_id, _box_indices_id])
		for data_id in +self.data_ids:
			llgr.delete_buffer(data_id)
		self.data_ids.clear()
		for matrix_id in +self.matrix_ids:
			llgr.delete_matrix(matrix_id)
		self.matrix_ids.clear()
		for object_id in self.object_ids:
			llgr.delete_object(object_id)
		self.object_ids.clear()
		llgr.delete_group(self.group_id)
		self.group_id = None
		track.modified(Graphics, [self], self.LESS_OBJECTS)

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
