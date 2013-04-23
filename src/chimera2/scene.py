"""
scene: scene management
=======================

The scene module is placeholder for demostration purposes.
A real scene module would have camera, lights, and more.

.. py:data:: bbox

   The current bounding box.  See :py:class:`BBox`.

"""

__all__ = [
	'BBox', 'bbox', 
	'reset',
	'add_sphere',
	'add_cylinder',
	'render'
]

from .math3d import Point, Vector, Translation, frustum, look_at, weighted_point, cross
from numpy import array

# attribute names
Position = "gl_Vertex"
Normal = "normal"

class BBox:
	"""axis-aligned bounding box

	If either :py:attr:`BBox.llf` or :py:attr:`BBox.urb` are None,
	then the bounding box is uninitialized.
	"""

	__slots__ = ['llf', 'urb']

	def __init__(self):
		self.llf = None	#: lower-left-front corner coordinates
		self.urb = None	#: upper-right-back corner coordinates

	def add_point(self, pt):
		"""extend bounding box to include given point

		:param pt: a :py:class:`~chimera2.math3d.Point`
		"""
		if self.llf is None:
			self.llf = Point(pt)
			self.urb = Point(pt)
			return
		for i in range(3):
			if pt[i] < self.llf[i]:
				self.llf[i] = pt[i]
			elif pt[i] > self.urb[i]:
				self.urb[i] = pt[i]

	def center(self):
		"""return center of bounding box
		
		:rtype: a :py:class:`~chimera2.math3d.Point`
		"""

		if self.llf is None:
			raise ValueError("empty bounding box")
		return weighted_point([self.llf, self.urb])

	def size(self):
		"""return length of sides of bounding box
		
		:rtype: a :py:class:`~chimera2.math3d.Vector`
		"""
		if self.llf is None:
			raise ValueError("empty bounding box")
		return self.urb - self.llf

bbox = BBox()
_program_id = 0

def reset():
	"""reinitialze scene
	
	Removes all objects, resets lights, bounding box,
	viewing transformation.
	"""
	global bbox, _program_id
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

def add_sphere(radius, center, color):
	"""add sphere to scene
	
	:param radius: the radius of the sphere
	:param center: the center of the sphere, :py:class:`~chimera2.math3d.Point`
	:param color: the RGBA color of the sphere (either a sequence of 4 floats, or an integer referring to a previously defined color)
	"""
	import llgr
	bbox.add_point(center - radius)
	bbox.add_point(center + radius)
	if isinstance(color, int):
		data_id = color
	else:
		data_id = llgr.next_data_id()
		assert len(color) == 4
		rgba = array(color, dtype='f').tostring()
		llgr.create_singleton(data_id, rgba)

	matrix_id = llgr.next_matrix_id()
	tmat = Translation(center)
	mat = array(tmat._matrix, dtype='f')
	llgr.create_matrix(matrix_id, mat, False)

	obj_id = llgr.next_object_id()
	ai = llgr.AttributeInfo("color", data_id, 0, 0, 4, llgr.Float)
	llgr.add_sphere(obj_id, radius, _program_id, matrix_id,
			[ai], Position, Normal)

def add_cylinder(radius, p0, p1, color):
	"""add cylinder to scene
	
	:param radius: the radius of the cylinder
	:param p0: one endpoint of the cylinder, :py:class:`~chimera2.math3d.Point`
	:param p1: the other endpoint of the cylinder, :py:class:`~chimera2.math3d.Point`
	:param color: the RGBA color of the cylinder (either a sequence of 4 floats, or an integer referring to a previously defined color)
	"""
	bbox.add_point(p0 - radius)
	bbox.add_point(p0 + radius)
	bbox.add_point(p1 - radius)
	bbox.add_point(p1 + radius)
	import llgr, math
	if isinstance(color, int):
		data_id = color
	else:
		data_id = llgr.next_data_id()
		assert len(color) == 4
		rgba = array(color, dtype='f').tostring()
		llgr.create_singleton(data_id, rgba)

	# create translation matrix
	matrix_id = llgr.next_matrix_id()
	cmat = Translation(weighted_point([p0, p1]))
	delta = p1 - p0
	height = delta.length()
	cylAxis = Vector([0, 1, 0])
	axis = cross(cylAxis, delta)
	cosine = (cylAxis * delta) / height
	angle = math.acos(cosine)
	if axis.sqlength() > 0:
		cmat.rotate(axis, angle)
	elif cosine < 0:	# delta == -cylAxis
		cmat.rotate(1, 0, 0, 180)
	mat = array(cmat._matrix, dtype='f')
	llgr.create_matrix(matrix_id, mat, False)

	obj_id = llgr.next_object_id()
	ai = llgr.AttributeInfo("color", data_id, 0, 0, 4, llgr.Float)
	llgr.add_cylinder(obj_id, radius, height, _program_id, matrix_id, [ai],
			Position, Normal)

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
	one = array([1, 1, 1, 1], dtype='f')
	ambient = array([0.197, 0.197, 0.197, 1], dtype='f')
	diffuse0 = array([0.432, 0.432, 0.432, 1], dtype='f')
	position0 = array([0.251, 0.251, 0.935, 0], dtype='f')
	diffuse1 = array([0.746, 0.746, 0.746, 1], dtype='f')
	position1 = array([-0.357, 0.66, 0.66, 0], dtype='f')
	shininess = array([30], dtype='f')
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

	if bbox.llf is not None:
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
