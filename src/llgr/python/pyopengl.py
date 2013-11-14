"""
PyOpenGL implementation of llgr.

The basic llgr interface is extended to manage Ids.

Add typechecking to function arguments to protect against arguments with the
wrong type being decoded by matching backend in JavaScript.
"""
import numpy
from typecheck import typecheck, Checker, either, list_of, TypeCheckError
from OpenGL import GL
from chimera2 import math3d
import weakref
import itertools
from .shader import ShaderProgram, ShaderVariable
from . import spiral

class Array(Checker):
	"""type annotation for numpy arrays"""

	def __init__(self, shape=None, dtype=None):
		self._shape = shape
		self._dtype = numpy.dtype(dtype)

	def __eq__(self, other):
		if self.__class__ is not other.__class__:
			return False
		return self._shape == other._shape and self._dtype == other._dtype

	def __hash__(self):
		return hash(str(hash(self.__class__)) + str(hash(self._shape)) + str(hash(self._dtype)))

	def __repr__(self):
		return "IsArray(%s, %s)" % (self._shape, self._dtype)

	def check(self, value):
		try:
			if not isinstance(value, numpy.ndarray):
				raise RuntimeError
			if self._shape and value.shape != self._shape:
				raise RuntimeError
			if self._dtype and value.dtype != self._dtype:
				raise RuntimeError
			return True
		except RuntimeError:
			msg = ''
			if self._shape:
				msg = 'x'.join(str(d) for d in self._shape)
			if self._dtype:
				if msg:
					msg += ' '
				msg += str(self._dtype)
			if msg:
				msg = "a %s array" % msg
			else:
				msg = "an array"
			raise TypeError(msg)

class _IsBuffer(Checker):
	"""type annotation for Python buffers"""

	def __repr__(self):
		return "IsBuffer"

	def check(self, value):
		try:
			memoryview(value)
		except TypeError:
			return False
		return True
IsBuffer = _IsBuffer()

class _GLint(Checker):
	"""type annotation for OpenGL GLint"""

	def __repr__(self):
		return "GLint"

	def check(self, value):
		return isinstance(value, int) and value >= 0 and value < 2147483648
GLint = _GLint()

class _GLclampf(Checker):
	"""type annotation for OpenGL GLclampf"""

	def __repr__(self):
		return "GLclampf"

	def check(self, value):
		return isinstance(value, (int, float)) and 0 <= value <= 1
GLclampf = _GLclampf()

class _nonnegative(Checker):
	"""type annonation for non-negative integers"""

	def __repr__(self):
		return "NonNeg32"

	def check(self, value):
		return isinstance(value, int) and 0 <= value < 2147483648
NonNeg32 = _nonnegative()

Number = either(int, float)

Id = NonNeg32		# negative Ids are reserved for llgr internal use

class Enum(Checker):
	"""subclass for C++-style enumerated type
	
	C++-style enumerated types' constants are in the same scope
	as the declaration.
	Also supports type annotations.
	"""

	values = ()
	labels = ()
	_initialized = set()
	_as_string = {}

	def __init__(self, value):
		if self.__class__ not in Enum._as_string:
			self.class_init()
		self.value = value

	@classmethod
	def class_init(cls, as_string=False):
		# inject enum constants into modeule's globals
		Enum._as_string[cls] = as_string
		if cls not in Enum._initialized:
			g = globals()
			for l, v in zip(cls.labels, cls.values):
				g[l] = cls(v)
			Enum._initialized.add(cls)

	def __repr__(self):
		if Enum._as_string[self.__class__]:
			return self.labels[self.value]
		else:
			return str(self.value)

	def __lt__(self, right):
		if not isinstance(right, self.__class__):
			raise ValueError("can not compare %s vs. %s" % (self.__class__.__name__, right.__class__.__name__))
		return self.value < right.value

	def check(self, value):
		if value not in self.values:
			return False
		return True

class DataType(Enum):
	values = list(range(7))
	labels = ('Byte', 'UByte', 'Short', 'UShort', 'Int', 'UInt', 'Float')
DataType.class_init()

class ShaderType(Enum):
	values = list(range(21))
	labels = ('IVec1', 'IVec2', 'IVec3', 'IVec4',
		'UVec1', 'UVec2', 'UVec3', 'UVec4',
		'FVec1', 'FVec2', 'FVec3', 'FVec4',
		'Mat2x2', 'Mat3x3', 'Mat4x4',
		'Mat2x3', 'Mat3x2', 'Mat2x4',
		'Mat4x2', 'Mat3x4', 'Mat4x3')
ShaderType.class_init()

#
# Program support
#

_pick_fragment_shader = """
#version 150

in vec3 f_pickId;
out vec3 frag_color;

void main (void)
{
  frag_color = f_pickId;
}
"""

_all_programs = {}
_pick_programs = {}

def create_program(program_id: Id, vertex_shader: str, fragment_shader: str, pick_vertex_shader: str):
	assert(program_id > 0)
	sp = ShaderProgram(vertex_shader, fragment_shader, "position")
	delete_program(program_id)
	_all_programs[program_id] = sp
	if not pick_vertex_shader:
		return
	sp = ShaderProgram(pick_vertex_shader, _pick_fragment_shader, "position")
	_pick_programs[program_id] = sp

def delete_program(program_id: Id):
	sp = _all_programs.get(program_id, None)
	if sp is None:
		return
	del _all_programs[program_id]
	sp.close()
	# TODO: invalidate vao and single_cache for objects that refer to this program
	sp = _pick_programs.get(program_id, None)
	if sp is None:
		return
	del _pick_programs[program_id]
	sp.close()

def clear_programs():
	programs = _all_programs
	_all_programs.clear()
	for sp in programs.values():
		sp.close()
	programs = _pick_programs
	_pick_programs.clear()
	for sp in programs.values():
		sp.close()

def _set_uniform(sv, shader_type, data):
	mv = memoryview(data)
	assert(mv.nbytes == sv.byte_count())
	if shader_type in (IVec1, IVec2, IVec3, IVec4, UVec1, UVec2, UVec3, UVec4):
		sv.set_intv(data)
	else:
		sv.set_floatv(data)

def set_uniform(program_id: Id, name: str, shader_type: ShaderType, data: IsBuffer):
	if shader_type > Mat2x2:
		set_uniform_matrix(program_id, name, False, shader_type, data)
		return
	if program_id == 0:
		# broadcast to all current programs
		for pid, sp in _all_programs.items():
			set_uniform(pid, name, shader_type, data)
		for pid, sp in _pick_programs.items():
			set_uniform(pid, name, shader_type, data)
		return
	sp = _all_programs.get(program_id, None)
	if sp is None:
		return
	sv = sp.uniform(name)
	if sv:
		_set_uniform(sv, shader_type, data)
	sp = _pick_programs.get(program_id, None)
	if sp is None:
		return
	sv = sp.uniform(name)
	if sv:
		_set_uniform(sv, shader_type, data)

def _set_uniform_matrix(sv, transpose, shader_type, data):
	mv = memoryview(data)
	assert(shader_type in (Mat2x2, Mat3x3, Mat4x4, Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3))
	assert(mv.nbytes == sv.byte_count())
	sv.set_float_matrixv(transpose, data)

def set_uniform_matrix(program_id: Id, name: str, transpose: bool, shader_type: ShaderType, data: IsBuffer):
	if program_id == 0:
		# broadcast to all current programs
		for pid, sp in _all_programs.items():
			set_uniform_matrix(pid, name, transpose, shader_type, data)
		for pid, sp in _pick_programs.items():
			set_uniform_matrix(pid, name, transpose, shader_type, data)
		return
	sp = _all_programs.get(program_id, None)
	if sp is None:
		return
	sv = sp.uniform(name)
	if sv:
		_set_uniform_matrix(sv, transpose, shader_type, data)
	sp = _pick_programs.get(program_id, None)
	if sp is None:
		return
	sv = sp.uniform(name)
	if sv:
		_set_uniform_matrix(sv, transpose, shader_type, data)

#
# (interleaved) buffer support
#

class _BufferInfo:

	def __init__(self, buffer, target, data=None):
		self.buffer = buffer
		self.target = target
		self.data = data	# only if singleton

_internal_buffer_id = itertools.count(start=-1, step=-1)

class BufferTarget(Enum):
	values = (GL.GL_ARRAY_BUFFER, GL.GL_ELEMENT_ARRAY_BUFFER)
	labels = ('ARRAY', 'ELEMENT_ARRAY')
BufferTarget.class_init()

_all_buffers = {}
_identity4x4_data = numpy.eye(4, dtype=numpy.float32)

# Create buffer of array data
def create_buffer(data_id: Id, buffer_target: BufferTarget, data: IsBuffer):
	if not _all_buffers:
		_all_buffers[0] = _BufferInfo(None, GL.GL_ARRAY_BUFFER, _identity4x4_data)
	if data_id in _all_buffers:
		buffer = _all_buffers[data_id].buffer
	else:
		buffer = GL.glGenBuffers(1)
	_all_buffers[data_id] = _BufferInfo(buffer, buffer_target)
	mv = memoryview(data)
	GL.glBindBuffer(buffer_target.value, buffer)
	GL.glBufferData(buffer_target.value, mv.nbytes, data, GL.GL_STATIC_DRAW)
	GL.glBindBuffer(buffer_target.value, 0)

def delete_buffer(data_id: Id):
	bi = _all_buffers.get(data_id, None)
	if bi is None:
		return
	del _all_buffers[data_id]
	if bi.buffer:
		GL.glDeleteBuffers(1, [bi.buffer])

def clear_buffers():
	buffers = _all_buffers
	_all_buffers.clear()
	for bi in buffers:
		if bi.buffer:
			GL.glDeleteBuffers(1, [bi.buffer])
	global _internal_buffer_id
	_internal_buffer_id = itertools.count(start=-1, step=-1)
	# clear internal data structures that created buffers
	clear_primitives()
	clear_matrices()

# create singleton "buffer" data
def create_singleton(data_id: Id, data: IsBuffer):
	if not _all_buffers:
		_all_buffers[0] = _BufferInfo(None, GL.GL_ARRAY_BUFFER, _identity4x4_data)
	if data_id in _all_buffers:
		buffer = _all_buffers[data_id].buffer
		GL.glDeleteBuffers(1, [buffer])
	_all_buffers[data_id] = _BufferInfo(None, GL.GL_ARRAY_BUFFER, data)

"""
# TODO
def create_singleton_index(data_id: Id, reference_data_id: Id, size: NonNeg32, offset: NonNeg32):
	pass # TODO

# update column of existing buffer data
def update_buffer(data_id: Id, offset: NonNeg32, stride: NonNeg32, count: NonNeg32, data: IsBuffer):
	pass # TODO
"""

# TODO: textures and volume support
#
# enum TextureFormat
class TextureFormat(int): pass
(RGB, RGBA, Luminance, LuminanceAlpha) = [TextureFormat(i) for i in range(4)]
# enum TextureFilter
class TextureFilter(int): pass
(Nearest, Linear, NearestMimapNearest, NearestMipmapNearest,
LinearMimapNearest, LinearMipmapLinear) = [TextureFilter(i) for i in range(6)]

def create_2d_texture(tex_id: Id, texture_format: TextureFormat, texture_min_filter: TextureFilter, texture_max_filter: TextureFilter, data_type: DataType, width: NonNeg32, height: NonNeg32, data: IsBuffer):
	pass # TODO

def create_3d_texture(tex_id: Id, texture_format: TextureFormat, texture_min_filter: TextureFilter, texture_max_filter: TextureFilter, data_type: DataType, width: NonNeg32, height: NonNeg32, depth: NonNeg32, data: IsBuffer):
	pass # TODO

def delete_texture(data_id: Id):
	pass # TODO

def clear_textures():
	pass

#
# matrices
#

class _MatrixInfo:

	def __init__(self, data_id, renormalize):
		self.data_id = data_id
		self.renormalize = renormalize

_all_matrices = {}
Matrix_4x4 = "4x4 float32 matrix"

# matrix_id of zero is reserved for identity matrix
# renormalize should be true when the rotation part of the matrix
# has shear or scaling, or if it is a projection matrix. 
def create_matrix(matrix_id: Id, matrix_4x4: Matrix_4x4, renormalize: bool=False):
	if isinstance(matrix_4x4, math3d.Xform):
		matrix_4x4 = matrix_4x4.getWebGLMatrix()
	else:
		assert(isinstance(matrix_4x4, numpy.ndarray))
	data_id = next(_internal_buffer_id)
	create_singleton(data_id, matrix_4x4)
	_all_matrices[matrix_id] = _MatrixInfo(data_id, renormalize)

def delete_matrix(matrix_id: Id):
	mi = _all_matrices.get(matrix_id, None)
	if mi is None:
		return
	del _all_matrices[matrix_id]
	delete_buffer(mi.data_id)

def clear_matrices():
	if not _all_buffers:
		# inside clear_buffers
		_all_matrices.clear()
		return
	matrices = _all_matrices
	_all_matrices.clear()
	for mi in matrices.values():
		delete_buffer(mi.data_id)

#
# Object support
#

class _ObjectInfo:

	def __init__(self, program_id, matrix_id, list_of_attribute_info, primitive_type, first, count, index_buffer_id=0, index_buffer_type=UByte):
		self.program_id = program_id
		self.matrix_id = matrix_id
		self.ais = list_of_attribute_info
		self.ptype = primitive_type
		self.first = first
		self.count = count
		self.index_buffer_id = index_buffer_id
		self.index_buffer_type = index_buffer_type
		self.hide = False
		self.transparent = False
		self.selected = False
		self.singleton_cache = None
		self.vao = None

_all_objects = {}

class AttributeInfo:

	def __init__(self, name: str, data_id: Id, offset: NonNeg32, stride: NonNeg32, count: NonNeg32, data_type: DataType, norm: bool=False):
		self.name = name
		self.data_id = data_id
		self.offset = offset
		self.stride = stride
		self.count = count
		self.data_type = data_type
		self.normalized = norm

	def __repr__(self):
		return 'AttributeInfo("%s", %r, %r, %r, %r, %r, %r)' % (
			self.name, self.data_id, self.offset, self.stride,
			self.count, self.data_type, self.normalized)

	def json(self):
		return (self.name, self.data_id, self.offset, self.stride, self.count, self.data_type, self.normalized)

class _SingletonInfo:

	def __init__(self, data_type, normalized, data, location, num_locations, num_elements):
		self.type = data_type
		self.normalized = normalized
		self.data = data
		self.base_location = location
		self.num_locations = num_locations
		self.num_elements = num_elements

# enum PrimitiveType
class PrimitiveType(Enum):
	values = (GL.GL_POINTS, GL.GL_LINES, GL.GL_LINE_LOOP, GL.GL_LINE_STRIP,
		GL.GL_TRIANGLES, GL.GL_TRIANGLE_STRIP, GL.GL_TRIANGLE_FAN)
	labels = ('Points', 'Lines', 'Line_loop', 'Line_strip',
			'Triangles', 'Triangle_strip', 'Triangle_fan')
PrimitiveType.class_init()

def _check_attributes(obj_id, oi):
	import sys
	sp = _all_programs.get(oi.program_id, None)
	if sp is None:
		print("missing program for object", obj_id, file=sys.stderr)
		return
	oi.singleton_cache = []
	for sv in sp.attributes:
		if sv.name == "instanceTransform":
			continue
		for ai in oi.ais:
			if ai.name == sv.name:
				break
		else:
			print("missing attribute", sv.name, "in object", obj_id, file=sys.stderr)
			continue
		bi = _all_buffers.get(ai.data_id, None)
		if bi is None:
			continue
		num_locations, num_elements = sv.location_info()
		if bi.data is not None:
			oi.singleton_cache.append(_SingletonInfo(ai.data_type,
				ai.normalized, bi.data, sv.location,
				num_locations, num_elements))
		else:
			_setup_array_attribute(bi, ai, sv.location, num_locations)
	if oi.index_buffer_id:
		ibi = _all_buffers.get(oi.index_buffer_id, None)
		if ibi is not None:
			GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibi.buffer)

def create_object(obj_id: Id, program_id: Id, matrix_id: Id,
		list_of_attributeInfo: list_of(AttributeInfo),
		primitive_type: PrimitiveType, first: NonNeg32, count: NonNeg32,
		index_data_id: Id=0, index_buffer_type: DataType=UByte):
	if index_data_id and index_buffer_type not in (UByte, UShort, UInt):
		raise ValueError("index_buffer_type must be unsigned")
	oi = _ObjectInfo(program_id, matrix_id, list_of_attributeInfo,
			primitive_type, first, count,
			index_data_id, index_buffer_type)
	delete_object(obj_id)
	try:
		oi.vao = GL.glGenVertexArrays(1)
		GL.glBindVertexArray(oi.vao)
		_check_attributes(obj_id, oi)
	finally:
		GL.glBindVertexArray(0)
	_all_objects[obj_id] = oi

def delete_object(obj_id: Id):
	if obj_id not in _all_objects:
		return
	del _all_objects[obj_id]

def clear_objects():
	_all_objects.clear()
	clear_groups()

# indicate whether to draw object or not
def hide_objects(list_of_objects: list_of(Id)):
	for obj_id in list_of_objects:
		_all_objects[obj_id].hide = True

def show_objects(list_of_objects: list_of(Id)):
	for obj_id in list_of_objects:
		_all_objects[obj_id].hide = False

# indicate whether an object is transparent or opaque (default opaque)
def transparent(list_of_objects: list_of(Id)):
	for obj_id in list_of_objects:
		_all_objects[obj_id].transparent = True

def opaque(list_of_objects: list_of(Id)):
	for obj_id in list_of_objects:
		_all_objects[obj_id].transparent = False

def selection_add(list_of_objects: list_of(Id)):
	for obj_id in list_of_objects:
		_all_objects[obj_id].selected = True

def selection_remove(list_of_objects: list_of(Id)):
	for obj_id in list_of_objects:
		_all_objects[obj_id].selected = False

def selection_clear():
	for obj_id in _all_objects:
		_all_objects[obj_id].selected = False

_all_groups = {}

def create_group(group_id: Id, objs: list_of(Id)):
	_all_groups[group_id] = objs

def delete_group(group_id: Id, and_objects: bool=False):
	if and_objects:
		objs = _all_groups[group_id]
		for obj in objs:
			delete_object(obj)
	del _all_groups[group_id]

def clear_groups(and_objects: bool=False):
	if _all_objects and and_objects:
		for objs in _all_groups.values():
			for obj in objs:
				delete_object(obj)
	_all_groups.clear()

def group_add(group_id: Id, obj_id: Id):
	objs = _all_groups[group_id]
	objs.append(obj_id)

def hide_group(group_id: Id):
	objs = _all_groups[group_id]
	hide_objects(objs)

def show_group(group_id: Id):
	objs = _all_groups[group_id]
	show_objects(objs)

def selection_add_group(group_id: Id):
	objs = _all_groups[group_id]
	selection_add(objs)

def selection_remove_group(group_id: Id):
	objs = _all_groups[group_id]
	selection_remove(objs)

def clear_all():
	clear_objects()
	clear_buffers()
	clear_programs()

# TODO: text primitives

#
# LOD primitives
#

class _PrimitiveInfo:

	def __init__(self, data_id, index_count, index_buffer_id, index_type):
		self.data_id = data_id
		self.icount = index_count
		self.index_id = index_buffer_id
		self.index_type = index_type

_proto_spheres = {}
_proto_cylinders = {}

def add_sphere(obj_id: Id, radius: Number, program_id: Id, matrix_id: Id,
			list_of_attribute_info: list_of(AttributeInfo)):
	num_pts = 100	# TODO: for LOD, make dependent on radius in pixels
	if num_pts not in _proto_spheres:
		_build_sphere(num_pts)
	pi = _proto_spheres[num_pts]
	mai = list_of_attribute_info[:]
	mai.append(AttributeInfo("normal", pi.data_id, 0, 12, 3, Float))
	mai.append(AttributeInfo("position", pi.data_id, 0, 12, 3, Float))
	scale_id = next(_internal_buffer_id)
	scale = numpy.array([radius, radius, radius], dtype=numpy.float32)
	create_singleton(scale_id, scale)
	mai.append(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float))
	create_object(obj_id, program_id, matrix_id, mai, Triangles, 0,
					pi.icount, pi.index_id, pi.index_type)

def _build_sphere(N):
	pts, phis, _ = spiral.points(N)
	tris = spiral.triangles(phis)
	data_id = next(_internal_buffer_id)
	create_buffer(data_id, ARRAY, pts)
	index_id = next(_internal_buffer_id)
	create_buffer(index_id, ELEMENT_ARRAY, tris)
	if len(tris) < 256:
		index_type = UByte
	elif len(tris) < 65536:
		index_type = UShort
	else:
		index_type = UInt
	_proto_spheres[N] = _PrimitiveInfo(data_id, tris.size, index_id, index_type)

def add_cylinder(obj_id: Id, radius: Number, length: Number,
			program_id: Id, matrix_id: Id,
			list_of_attribute_info: list_of(AttributeInfo)):
	num_pts = 40	# TODO: for LOD, make depending on radius in pixels
	if num_pts not in _proto_cylinders:
		_build_cylinder(num_pts)
	pi = _proto_cylinders[num_pts]
	mai = list_of_attribute_info[:]
	mai.append(AttributeInfo("normal", pi.data_id, 0, 24, 3, Float))
	mai.append(AttributeInfo("position", pi.data_id, 12, 24, 3, Float))
	scale_id = next(_internal_buffer_id)
	scale = numpy.array([radius, length / 2., radius], dtype=numpy.float32)
	create_singleton(scale_id, scale)
	mai.append(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float))
	create_object(obj_id, program_id, matrix_id, mai, Triangle_strip, 0,
					pi.icount, pi.index_id, pi.index_type)

def _build_cylinder(N):
	from math import sin, cos, pi
	# normal & position array
	np = numpy.zeros((N * 2, 6), dtype=numpy.float32)
	num_indices = N * 2 + 2
	indices = numpy.zeros(num_indices, dtype=numpy.uint16)
	for i in range(N):
		theta = 2 * pi * i / N
		x = cos(theta)
		z = sin(theta)
		np[i][0] = x;	# nx
		np[i][1] = 0;	# ny
		np[i][2] = z;	# nz
		np[i][3] = x;	# px
		np[i][4] = -1;	# py
		np[i][5] = z;	# pz
		np[i + N] = np[i]
		np[i + N][4] = 1
		indices[i * 2] = i
		indices[i * 2 + 1] = i + N
	indices[N * 2] = 0
	indices[N * 2 + 1] = N

	data_id = next(_internal_buffer_id)
	create_buffer(data_id, ARRAY, np)
	index_id = next(_internal_buffer_id)
	create_buffer(index_id, ELEMENT_ARRAY, indices)
	_proto_cylinders[N] = _PrimitiveInfo(data_id, num_indices, index_id, UShort)

def _clear_geom(geom):
	if not _all_buffers:
		for g in geom.values():
			delete_buffer(g.data_id)
			delete_buffer(g.index_id)
	geom.clear()

def clear_primitives():
	_clear_geom(_proto_spheres)
	_clear_geom(_proto_cylinders)

#
# rendering
#

_clear_color = [0.0, 0.0, 0.0, 1.0]

def set_clear_color(red: GLclampf, green: GLclampf, blue: GLclampf, alpha: GLclampf):
	global _clear_color
	_clear_color = [red, green, blue, alpha]
	GL.glClearColor(red, green, blue, alpha);

_data_type_map = {
	Byte: GL.GL_BYTE,
	UByte: GL.GL_UNSIGNED_BYTE,
	Short: GL.GL_SHORT,
	UShort: GL.GL_UNSIGNED_SHORT,
	Int: GL.GL_INT,
	UInt: GL.GL_UNSIGNED_INT,
	Float: GL.GL_FLOAT,
}

def _cvt_data_type(data_type):
	# return GLenum corresponding to data type
	return _data_type_map.get(data_type, GL.GL_NONE)

_data_size_map = {
	Byte: 1,
	UByte: 1,
	Short: 2,
	UShort: 2,
	Int: 4,
	UInt: 4,
	Float: 4,
}

def _data_size(data_type):
	# return size in bytes of data type
	return _data_size_map.get(data_type, 0)

def _setup_array_attribute(bi, ai, loc, num_locations):
	gl_type = _cvt_data_type(ai.data_type)
	GL.glBindBuffer(bi.target.value, bi.buffer)
	# TODO: if shader variable is int, use glVertexAttribIPointer
	import ctypes
	# Pointer arg must be void_p, not an integer.
	offset = ctypes.c_void_p(ai.offset)
	GL.glVertexAttribPointer(loc, ai.count, gl_type, ai.normalized,
			ai.stride, offset)
	for i in range(loc, loc + num_locations):
		GL.glEnableVertexAttribArray(i)

_did_once = False

_singleton_map = {
	1: {
		Short: GL.glVertexAttrib1sv,
		UShort: GL.glVertexAttrib1sv,
		Float: GL.glVertexAttrib1fv,
	},
	2: {
		Short: GL.glVertexAttrib2sv,
		UShort: GL.glVertexAttrib2sv,
		Float: GL.glVertexAttrib2fv,
	},
	3: {
		Short: GL.glVertexAttrib3sv,
		UShort: GL.glVertexAttrib3sv,
		Float: GL.glVertexAttrib3fv,
	},
	4: {
		Byte: GL.glVertexAttrib4bv,
		UByte: GL.glVertexAttrib4ubv,
		Short: GL.glVertexAttrib4sv,
		UShort: GL.glVertexAttrib4usv,
		Int: GL.glVertexAttrib4iv,
		UInt: GL.glVertexAttrib4uiv,
		Float: GL.glVertexAttrib4fv,
	},
	5: {
		Byte: GL.glVertexAttrib4Nbv,
		UByte: GL.glVertexAttrib4Nubv,
		Short: GL.glVertexAttrib4Nsv,
		UShort: GL.glVertexAttrib4Nusv,
		Int: GL.glVertexAttrib4Niv,
		UInt: GL.glVertexAttrib4Nuiv,
	},
}

def _setup_singleton_attribute(data, data_type, normalized, loc, num_locations, num_elements):
	if data_type != Float:
		global _did_once
		if not _did_once:
			import sys
			print("WebGL only supports float singleton vertex attributes\n", file=sys.stderr)
			_did_once = True

	if num_locations > 1:
		data = bytes(data)
	size = num_elements * _data_size(data_type)
	if num_elements == 4 and normalized:
		num_elements = 5
	for i in range(num_locations):
		func = _singleton_map[num_elements].get(data_type, None)
		if func:
			func(loc + i, data[i * size:(i + 1) * size])

_darwin_vao = None

def render():
	import sys
	global _darwin_vao
	if sys.platform == 'darwin' and _darwin_vao is None:
		# using glVertexAttribPointer fails unless a VAO is bound
		# even if VAO's aren't used
		_darwin_vao = GL.glGenVertexArrays(1)
		GL.glBindVertexArray(_darwin_vao)

	GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT|GL.GL_STENCIL_BUFFER_BIT)
	GL.glEnable(GL.GL_DEPTH_TEST)
	GL.glEnable(GL.GL_DITHER)
	GL.glDisable(GL.GL_SCISSOR_TEST)

	if not _all_objects:
		return

	sp = None
	current_program_id = 0
	current_matrix_id = 2 ** 31
	# instance transform (it_) singleton info
	it_type = Float
	it_data = None
	it_loc = -1
	it_locations, it_elements = ShaderVariable.type_location_info(ShaderVariable.Mat4x4)
	# TODO: only for opaque objects
	GL.glEnable(GL.GL_CULL_FACE)
	GL.glDisable(GL.GL_BLEND)
	for oi in _all_objects.values():
		if oi.hide or not oi.program_id:
			continue
		# setup program
		if oi.program_id != current_program_id:
			new_sp = _all_programs.get(oi.program_id, None)
			if new_sp is None:
				continue
			new_sp.setup()
			sp = new_sp
			current_program_id = oi.program_id
			current_matrix_id = 2 ** 31
			sv = sp.attribute("instanceTransform")
			it_loc = sv.location
		if sp is None:
			continue
		GL.glBindVertexArray(oi.vao)
		# setup instance matrix attribute
		if oi.matrix_id != current_matrix_id:
			if oi.matrix_id == 0:
				data_id = 0
			else:
				mi = _all_matrices.get(oi.matrix_id, None)
				if mi is None:
					continue
				data_id = mi.data_id
			bi = _all_buffers.get(data_id, None)
			if bi is None:
				continue
			it_data = bi.data
			_setup_singleton_attribute(it_data, it_type, False, it_loc, it_locations, it_elements)
			current_matrix_id = oi.matrix_id

		for si in oi.singleton_cache:
			_setup_singleton_attribute(si.data, si.type, si.normalized, si.base_location, si.num_locations, si.num_elements)
		# finally draw object
		if oi.index_buffer_id == 0:
			GL.glDrawArrays(oi.ptype.value, oi.first, oi.count)
		else:
			import ctypes
			offset = ctypes.c_void_p(oi.first * _data_size(oi.index_buffer_type))
			GL.glDrawElements(oi.ptype.value, oi.count,
				_cvt_data_type(oi.index_buffer_type),
				offset)
	GL.glBindVertexArray(0)
	if sp:
		sp.cleanup()

class _PickId(bytearray):

	def __init__(self):
		bytearray.__init__(self, 3)

	def set(self, i):
		self[0] = i % 256
		self[1] = (i // 256) % 256
		self[2] = (i // 65536) % 256

def pick(x: int, y: int):
	# Just like rendering except the color is monotonically increasing
	# and varies by object, not within an object.
	# Assume WebGL defaults of 8-bits each for red, green, and blue,
	# for a maximum of 16,777,215 (2^24 - 1) objects and that object ids
	# are also less than 16,777,215.

	import sys
	global _darwin_vao
	if sys.platform == 'darwin' and _darwin_vao is None:
		# using glVertexAttribPointer fails unless a VAO is bound
		# even if VAO's aren't used
		_darwin_vao = GL.glGenVertexArrays(1)
		GL.glBindVertexArray(_darwin_vao)

	# render 5x5 pixels around hot spot
	GL.glScissor(x - 2, y - 2, 5, 5)
	GL.glEnable(GL.GL_SCISSOR_TEST)

	GL.glClearColor(0, 0, 0, 0);
	GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT|GL.GL_STENCIL_BUFFER_BIT)
	GL.glEnable(GL.GL_DEPTH_TEST)
	GL.glDisable(GL.GL_DITHER)
	GL.glClearColor(*_clear_color)

	if not _all_objects:
		return 0

	sp = None
	current_program_id = 0
	current_matrix_id = 2 ** 31
	# instance transform (it_) singleton info
	it_type = Float
	it_data = None
	it_loc = -1
	it_locations, it_elements = ShaderVariable.type_location_info(ShaderVariable.Mat4x4)

	# pick id (pi_) singleton info
	pi_type = UByte
	pi_data = _PickId()
	pi_loc = 0
	pi_locations, pi_elements = ShaderVariable.type_location_info(ShaderVariable.Vec3)

	# TODO: only for opaque objects
	GL.glEnable(GL.GL_CULL_FACE)
	GL.glDisable(GL.GL_BLEND)
	for oi in _all_objects.values():
		if oi.hide or not oi.program_id:
			continue
		# setup program
		if oi.program_id != current_program_id:
			new_sp = _all_programs.get(oi.program_id, None)
			if new_sp is None:
				continue
			new_sp.setup()
			sp = new_sp
			current_program_id = oi.program_id
			current_matrix_id = 2 ** 31
			sv = sp.attribute("instanceTransform")
			it_loc = sv.location
		if sp is None:
			continue
		GL.glBindVertexArray(oi.vao)
		# setup instance matrix attribute
		if oi.matrix_id != current_matrix_id:
			if oi.matrix_id == 0:
				data_id = 0
			else:
				mi = _all_matrices.get(oi.matrix_id, None)
				if mi is None:
					continue
				data_id = mi.data_id
			bi = _all_buffers.get(data_id, None)
			if bi is None:
				continue
			it_data = bi.data
			_setup_singleton_attribute(it_data, it_type, False, it_loc, it_locations, it_elements)
			current_matrix_id = oi.matrix_id

		for si in oi.singleton_cache:
			_setup_singleton_attribute(si.data, si.type, si.normalized, si.base_location, si.num_locations, si.num_elements)
		# finally draw object
		if oi.index_buffer_id == 0:
			GL.glDrawArrays(oi.ptype.value, oi.first, oi.count)
		else:
			import ctypes
			offset = ctypes.c_void_p(oi.first * _data_size(oi.index_buffer_type))
			GL.glDrawElements(oi.ptype.value, oi.count,
				_cvt_data_type(oi.index_buffer_type),
				offset)
	GL.glBindVertexArray(0)
	if sp:
		sp.cleanup()


#
# vsphere
#

class _VSphereInfo:

	def _init__(self):
		self.radius = 0
		self.center = (0, 0)
		self.xy = (0, 0)
		self.cursor = None

_all_vspheres = {}

_FUZZY_ZERO = 1e-4
_FUZZY_SQZERO = 1e-8

class Cursors(Enum):
	values = list(range(2))
	labels = ('Rotation', 'ZRotation')
Cursors.class_init()

def _compute_vsphere(fx, fy, tx, ty):
	# return what, spin_axis, spin_angle
	from math import sqrt, acos
	d1 = fx * fx + fy * fy
	d2 = tx * tx + ty * ty

	if d1 > 1 and d2 < 1:
		# transition from z rotation to sphere rotation
		return 1, None, 0
	if d1 < 1 and d2 > 1:
		# transition from sphere rotation to z rotation
		return 2, None, 0
	if d1 < 1:
		from_ = math3d.Vector([fx, fy, sqrt(1 - d1)])
		to = math3d.Vector([tx, ty, sqrt(1 - d2)])
	else:
		d1 = sqrt(d1)
		d2 = sqrt(d2)
		from_ = math3d.Vector([fx / d1, fy / d1, 0])
		to = math3d.Vector([tx / d2, ty / d2, 0])
	spin_axis = math3d.cross(from_, to)
	if spin_axis.sqlength() < _FUZZY_SQZERO:
		# if the two positions normalized to the same vector, punt.
		return 3, None, 0
	dot_product = from_ * to;	# from and to are "unit" length
	if dot_product > 1:
		# guarantee within acos bounds (more of a problem with floats)
		dot_product = 1
	spin_angle = acos(dot_product)
	if -_FUZZY_ZERO < spin_angle < _FUZZY_ZERO:
		# may need to adjust threshold to avoid drift
		# on a per-input device basis
		return 4, None, 0
	return 0, spin_axis, spin_angle

def vsphere_setup(vsphere: Id, radius: float, center: (float, float)):
	vi = _all_vspheres.get(vsphere, None)
	if vi is None:
		vi = _all_vspheres[vsphere] = _VSphereInfo()
	vi.radius = radius
	vi.center = center
	# other fields are initialized in vsphere_press

def vsphere_press(vsphere: Id, x: int, y: int):
	vi = _all_vspheres.get(vsphere, None)
	if vi is None:
		raise ValueError("unknown vsphere")

	vi.xy = (x, y)
	tx = (x - vi.center[0]) / vi.radius
	ty = -(y - vi.center[1]) / vi.radius
	if (tx * tx + ty * ty >= 1):
		vi.cursor = ZRotation
	else:
		vi.cursor = Rotation
	return vi.cursor

def vsphere_drag(vsphere: Id, x: int, y: int, throttle: bool):
	vi = _all_vspheres.get(vsphere, None)
	if vi is None:
		raise ValueError("unknown vsphere")

	fx = (vi.xy[0] - vi.center[0]) / vi.radius
	fy = -(vi.xy[1] - vi.center[1]) / vi.radius
	tx = (x - vi.center[0]) / vi.radius
	ty = -(y - vi.center[1]) / vi.radius

	what, spin_axis, spin_angle = _compute_vsphere(fx, fy, tx, ty)
	if what == 0:
		# normal case: rotation
		vi.xy = x, y
	elif what == 1:
		# transition z-rotation to rotation
		vi.cursor = Rotation
		vi.xy = x, y
	elif what == 2:
		# transition rotation to z-rotation
		vi.cursor = ZRotation
		vi.xy = x, y
	elif what == 3:
		# from and to normalized to same point
		# don't update last x and y
		pass
	elif what == 4:
		# angle effectively zero
		# don't update last x and y
		pass
	if throttle:
		spin_angle *= 0.1
	return vi.cursor, spin_axis, spin_angle

def vsphere_release(vsphere: Id):
	if vsphere not in _all_vspheres:
		raise ValueError("unknown vsphere")
	del _all_vspheres[vsphere]
