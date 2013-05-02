"""
Dump llgr calls.  Emit when render() is called.

The basic llgr interface is extended to manage Ids.

Add typechecking to function arguments to protect against arguments with the
wrong type being decoded by matching backend in JavaScript.
"""
import numpy
import typecheck
from typecheck import accepts

JSON_FORMAT = 'json'
CPP_FORMAT = 'c++'
JS_FORMAT = 'js'

FORMATS = (JSON_FORMAT, CPP_FORMAT, JS_FORMAT)

_dump_format = JSON_FORMAT

class Array(typecheck.TypeAnnotation):
	"""type annotation for numpy arrays"""

	def __init__(self, shape=None, dtype=None):
		self.type = self
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

	def __typecheck__(self, func, to_check):
		try:
			if not isinstance(to_check, numpy.ndarray):
				raise RuntimeError
			if self._shape and to_check.shape != self._shape:
				raise RuntimeError
			if self._dtype and to_check.dtype != self._dtype:
				raise RuntimeError
			return
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
			raise typecheck._TC_TypeError(to_check, msg)

class _IsBuffer(typecheck.TypeAnnotation):
	"""type annotation for Python buffers"""

	def __init__(self):
		self.type = self

	def __eq__(self, other):
		return self.__class__ is other.__class__

	# They're all the same
	def __hash__(self):
		return id(self.__class__)

	def __repr__(self):
		return "IsBuffer"

	def __typecheck__(self, func, to_check):
		try:
			memoryview(to_check)
		except TypeError:
			raise typecheck._TC_TypeError(to_check, "a buffer")
IsBuffer = _IsBuffer()

class _nonnegative(typecheck.TypeAnnotation):
	"""type annonation for non-negative integers"""

	def __init__(self):
		self.type = self

	def __eq__(self, other):
		return self.__class__ is other.__class__

	# They're all the same
	def __hash__(self):
		return id(self.__class__)

	def __repr__(self):
		return "NonNeg"

	def __typecheck__(self, func, to_check):
		if not isinstance(to_check, int) or to_check < 0:
			raise typecheck._TC_TypeError(to_check, "non-negative integer")
NonNeg = _nonnegative()
Id = NonNeg		# negative Ids are reserved for llgr internal use

from typecheck.typeclasses import Number, String

class Enum(typecheck.TypeAnnotation):
	"""sublass for type annotation of enumerated type"""

	values = ()
	labels = ()

	def __init__(self):
		self.type = self
		self.reinit()

	def reinit(self, as_string=False):
		# inject enum constants into modeule's globals
		g = globals()
		if as_string:
			for l, v in zip(self.labels, self.values):
				g[l] = l
		else:
			for l, v in zip(self.labels, self.values):
				g[l] = v

	def __eq__(self, other):
		return self.__class__ is other.__class__

	# They're all the same
	def __hash__(self):
		return id(self.__class__)

	def __repr__(self):
		return self.__class__.__name__

	def __typecheck__(self, func, to_check):
		if _dump_format == JSON_FORMAT:
			if to_check not in self.values:
				raise typecheck._TC_TypeError(to_check, "enum")
		if to_check not in self.labels:
			raise typecheck._TC_TypeError(to_check, "enum")

# next line is for pyflakes
Byte, UByte, Short, UShort, Int, UInt, Float = 0, 0, 0, 0, 0, 0, 0
class _DataType(Enum):
	values = list(range(7))
	labels = ('Byte', 'UByte', 'Short', 'UShort', 'Int', 'UInt', 'Float')
DataType = _DataType()

# next three lines are for pyflakes
IVec1, IVec2, IVec3, IVec4 = 0, 0, 0, 0
UVec1, UVec2, UVec3, UVec4 = 0, 0, 0, 0
FVec1, FVec2, FVec3, FVec4 = 0, 0, 0, 0
Mat2x2, Mat3x3, Mat4x4 = 0, 0, 0
class _ShaderType(Enum):
	values = list(range(15))
	labels = ('IVec1', 'IVec2', 'IVec3', 'IVec4', 'UVec1', 'UVec2', 'UVec3', 'UVec4', 'FVec1', 'FVec2', 'FVec3', 'FVec4', 'Mat2x2', 'Mat3x3', 'Mat4x4')
	#Mat2x3, Mat3x2, Mat2x4,
	#Mat4x2, Mat3x4, Mat4x3)
ShaderType = _ShaderType()

def convert_buffer(b):
	"""return [little-endian, size, [unsigned integers]]"""
	b = buffer(b)
	if _dump_format != JSON_FORMAT:
		type = "missing"
		if _dump_format == CPP_FORMAT:
			type = "const char*"
		elif _dump_format == JS_FORMAT:
			type = "var"
		text = repr(bytes(b))[1:-1].replace('"', '\\"')
		return type, '"%s"' % text, len(b)
	import struct
	size = len(b)
	fmt = 'I' * (size // 4)
	fmt = '<' + fmt + ['', 'B', 'H', 'HB'][size % 4]
	return [True, size, struct.unpack(fmt, b)]

_calls = []

class save_args(object):
	"""Decorator that caches arguments in global _calls list"""

	def __init__(self, func):
		self.func = func

	def __call__(self, *args, **kw):
		import inspect
		if kw:
			# turn keyword arguments into positional arguments
			# for serialization
			spec = inspect.getargspec(self.func)
			if spec.varargs:
				raise ValueError("can not serialize function"
					" with infinite arguments and keywords")
			args = args[:]
			num_kw_args = len(spec.defaults)
			for i, n in enumerate(spec.args[-num_kw_args:]):
				if n in kw:
					args.append(kw[n])
				else:
					args.append(spec.defaults[i])
		if _dump_format == JSON_FORMAT:
			info = [self.func.__name__]
			if args or kw:
				info.append(args)
			_calls.append(info)
			self.func(*args)
		else:
			_calls.append('\t%s(%s);' % (self.func.__name__,
							repr(args)[1:-1]))

	def __repr__(self):
		return self.func.__doc__


def render():
	global _calls
	tmp = _calls
	_calls = []
	if _dump_format == JSON_FORMAT:
		import json
		with open("render.json", "w") as f:
			json.dump(tmp, f)
			print >> f	# trailing newline
		return
	if _dump_format == CPP_FORMAT:
		with open("render.cpp", "w") as f:
			print >> f, "#include <llgr.h>\n"
			print >> f, "#define False 0\n"
			print >> f, "#define True 1\n"
			print >> f, "using namespace llgr;\n"
			print >> f, "void\ninit_objects()\n{"
			for call in tmp:
				print >> f, '%s' % call
			print >> f, "}"
		return
	if _dump_format == JS_FORMAT:
		with open("render.js", "w") as f:
			print >> f, "function init_objects()\n{"
			for call in tmp:
				print >> f, "%s;" % call
			print >> f, "}"

@accepts(Id, String, String, String)
def create_program(program_id, vertex_shader, fragment_shader, pick_vertex_shader):
	if _dump_format == JSON_FORMAT:
		_calls.append(['create_program', [program_id, vertex_shader, fragment_shader]])
		return
	vs = repr(vertex_shader)[1:-1]
	vs = vs.replace('"', '\\"').replace('\\n', '\\n\\\n')
	fs = repr(fragment_shader)[1:-1]
	fs = fs.replace('"', '\\"').replace('\\n', '\\n\\\n')
	pvs = repr(fragment_shader)[1:-1]
	pvs = fs.replace('"', '\\"').replace('\\n', '\\n\\\n')
	_calls.append('\tcreate_program(%s, "%s", "%s", "%s");' % (program_id, vs, fs, pvs))
@save_args
@accepts(Id)
def delete_program(program_id):
	pass
@save_args
@accepts()
def clear_programs():
	pass

@accepts(Id, String, ShaderType, IsBuffer)
def set_uniform(program_id, name, shader_type, data):
	if _dump_format == JSON_FORMAT:
		_calls.append(['set_uniform', [program_id, name, shader_type, convert_buffer(data)]])
		return
	uname = 'u_%s%s' % (name, program_id)
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, uname, init))
	_calls.append('\tset_uniform(%s, "%s", %s, %s, %s);'
			% (program_id, name, shader_type, num_bytes, uname))

@accepts(Id, String, bool, ShaderType, IsBuffer)
def set_uniform_matrix(program_id, name, transpose, shader_type, data):
	if _dump_format == JSON_FORMAT:
		_calls.append(['set_uniform_matrix', [program_id, name, transpose, shader_type, convert_buffer(data)]])
		return
	uname = 'u_%s%s' % (name, program_id)
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, uname, init))
	_calls.append('\tset_uniform_matrix(%s, "%s", %s, %s, %s, %s);'
		% (program_id, name, transpose, shader_type, num_bytes, uname))

# (interleaved) buffer support

class _BufferTarget(Enum):
	values = (0x8892, 0x8893)
	labels = ('ARRAY', 'ELEMENT_ARRAY')
BufferTarget = _BufferTarget()

# Create buffer of array data
@accepts(Id, BufferTarget, IsBuffer)
def create_buffer(data_id, buffer_target, data):
	if _dump_format == JSON_FORMAT:
		_calls.append(['create_buffer', [data_id, buffer_target, convert_buffer(data)]])
		return
	dname = ('d%s' % data_id).replace('-', '_')
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, dname, init))
	_calls.append('\tcreate_buffer(%s, %s, %s, %s);'
				% (data_id, buffer_target, num_bytes, dname))
@save_args
@accepts(Id)
def delete_buffer(data_id):
	pass
@save_args
@accepts()
def clear_buffers():
	pass

# create singleton "buffer" data
@accepts(Id, IsBuffer)
def create_singleton(data_id, data):
	if _dump_format == JSON_FORMAT:
		_calls.append(['create_singleton', [data_id, convert_buffer(data)]])
		return
	dname = ('d%s' % data_id).replace('-', '_')
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, dname, init))
	_calls.append('\tcreate_singleton(%s, %s, %s);' % (data_id, num_bytes, dname))

"""
# TODO
@save_args
@accepts(Id, Id, NonNeg, NonNeg)
def create_singleton_index(data_id, reference_data_id, size, offset):
	pass # TODO

# update column of existing buffer data
@accepts(Id, NonNeg, NonNeg, NonNeg, IsBuffer):
def update_buffer(data_id, offset, stride, count, data):
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
@accepts(int, TextureFormat, TextureFilter, TextureFilter, DataType, NonNeg, NonNeg, IsBuffer)
def create_2d_texture(tex_id, texture_format, texture_min_filter, texture_max_filter, data_type, width, height, data):
	pass # TODO
@accepts(int, TextureFormat, TextureFilter, TextureFilter, DataType, NonNeg, NonNeg, NonNeg, IsBuffer)
def create_3d_texture(tex_id, texture_format, texture_min_filter, texture_max_filter, data_type, width, height, depth, data):
	pass # TODO
@save_args
@accepts(Id)
def delete_texture(data_id):
	pass # TODO
@save_args
@accepts()
def clear_textures():
	pass

# matrices

def set_projection_matrix(matrix_4x4, name="ProjectionMatrix"):
	# program_id of zero means all programs
	matrix_4x4 = numpy.array(list(matrix_4x4), dtype='f')
	bytes = matrix_4x4.tostring()
	set_uniform(0, name, Mat4x4, bytes)
def set_modelview_matrix(matrix_4x4, mv="ModelViewMatrix", normal="NormalMatrix"):
	# Matrix is assumed to be pure rotation and translation.
	# program_id of zero means all programs
	flat = list(matrix_4x4)
	matrix_4x4 = numpy.array(flat, dtype='f')
	bytes = matrix_4x4.tostring()
	set_uniform(0, mv, Mat4x4, bytes)
	# TODO: remove assumption that  rotation part is orthonormal
	matrix_3x3 = numpy.array(flat[0:3] + flat[4:7] + flat[8:11], dtype='f')
	bytes = matrix_3x3.tostring()
	set_uniform(0, normal, Mat3x3, bytes)

# type check for 4x4 array, dtype='f' (aka dtype('float32'))
Matrix_4x4 = typecheck.Or(Array(shape=(4,4), dtype='f'), (
	float, float, float, float,
	float, float, float, float,
	float, float, float, float,
	float, float, float, float
))

# matrix_id of zero is reserved for identity matrix
# renormalize should be true when the rotation part of the matrix
# has shear or scaling, or if it is a projection matrix. 
@accepts(Id, Matrix_4x4, bool)
def create_matrix(matrix_id, matrix_4x4, renormalize = False):
	if isinstance(matrix_4x4, numpy.ndarray):
		m = [float(f) for f in matrix_4x4.flat]
	else:
		m = list(matrix_4x4)
	if _dump_format == JSON_FORMAT:
		_calls.append(['create_matrix', [matrix_id, m, renormalize]])
		return
	m4x4 = tuple(zip(*([iter(m)] * 4)))
	mname = ('m%s' % matrix_id).replace('-', '_')
	mat = repr(m4x4).replace('(', '{').replace(')', '}')
	_calls.append('\tfloat %s[4][4] = %s;' % (mname, mat))
	_calls.append('\tcreate_matrix(%s, %s, %s);'
					% (matrix_id, mname, renormalize))
@save_args
@accepts(Id)
def delete_matrix(matrix_id):
	pass
@save_args
@accepts()
def clear_matrices():
	pass

# flat scene graph


#struct AttributeInfo {
#	std::string name;
#	Id	data_id;
#	uint32_t offset;	# byte offset into buffer
#	uint32_t stride;	# byte stride to next element in buffer
#	uint32_t count;		# number of data type
#	DataType type;
#	bool	normalized;	# only for integer types
#};
#typedef std::vector<AttributeInfo> AttributeInfos;

class _AttributeInfo(object):

	def __init__(self, name, data_id, offset, stride, count, data_type, norm=False):
		self.name = name
		self.data_id = data_id
		self.offset = offset
		self.stride = stride
		self.count = count
		self.data_type = data_type
		self.normalize = norm

	def __repr__(self):
		if _dump_format == JSON_FORMAT:
			return (self.name, self.data_id, self.offset, self.stride, self.count, self.data_type, self.norm)
		return 'AttributeInfo("%s", %r, %r, %r, %r, %r, %r)' % (
			self.name, self.data_id, self.offset, self.stride,
			self.count, self.data_type, self.normalize)

@accepts(String, Id, NonNeg, NonNeg, NonNeg, DataType, norm=bool)
def AttributeInfo(name, data_id, offset, stride, count, data_type, norm=False):
	return _AttributeInfo(name, data_id, offset, stride, count, data_type, norm)
_AttributeInfos = [_AttributeInfo]

# enum PrimitiveType
class _PrimitiveType(Enum):
	values = list(range(7))
	labels = ('Points', 'Lines', 'Line_loop', 'Line_strip',
			'Triangles', 'Triangle_strip', 'Triangle_fan')
PrimitiveType = _PrimitiveType()

@save_args
@accepts(Id, Id, Id, _AttributeInfos, PrimitiveType, NonNeg, NonNeg, index_data_id=Id, index_buffer_type=DataType)
def create_object(obj_id, program_id, matrix_id,
		list_of_attributeInfo, primitive_type,
		first, count,
		index_data_id = 0, index_buffer_type = UByte):
	pass
@save_args
@accepts(Id)
def delete_object(obj_id):
	pass
@save_args
@accepts()
def clear_objects():
	pass

# indicate whether to draw object or not
@save_args
@accepts([Id])
def hide_objects(list_of_objects):
	pass
@save_args
@accepts([Id])
def show_objects(list_of_objects):
	pass

# indicate whether an object is transparent or opaque (default opaque)
@save_args
@accepts([Id])
def transparent(list_of_objects):
	pass
@save_args
@accepts([Id])
def opaque(list_of_objects):
	pass

# TODO: text primitives

# LOD primitives

@save_args
@accepts(String, String)
def set_primitive_attribute_name(name, value):
	pass

@accepts(Id, float, Id, Id, _AttributeInfos)
def add_sphere(obj_id, radius, program_id, matrix_id, list_of_attributeInfo):
	if _dump_format == JSON_FORMAT:
		_calls.append(['add_sphere', [obj_id, radius, program_id, matrix_id, list_of_attributeInfo]])
		return
	aname = ('a%s' % obj_id).replace('-', '_')
	_calls.append('\tAttributeInfos %s;' % aname)
	for ai in list_of_attributeInfo:
		_calls.append("\t%s.push_back(%s);" % (aname, ai))
	_calls.append('\tadd_sphere(%r, %r, %r, %r, %s, "%s", "%s");'
			% (obj_id, radius, program_id, matrix_id, aname))

@accepts(Id, float, float, Id, Id, _AttributeInfos)
def add_cylinder(obj_id, radius, length,
		program_id, matrix_id, list_of_attributeInfo):
	if _dump_format == JSON_FORMAT:
		_calls.append(['add_cylinder', [obj_id, radius, length, program_id, matrix_id, list_of_attributeInfo]])
		return
	aname = ('a%s' % obj_id).replace('-', '_')
	_calls.append('\tAttributeInfos %s;' % aname)
	for ai in list_of_attributeInfo:
		_calls.append("\t%s.push_back(%s);" % (aname, ai))
	_calls.append('\tadd_cylinder(%r, %r, %r, %r, %r, %s, "%s", "%s");'
			% (obj_id, radius, length, program_id,
			matrix_id, aname))

# misc

@save_args
@accepts()
def clear_primitives():
	pass

@save_args
@accepts()
def clear_all():
	pass

@save_args
@accepts(Number, Number, Number, Number)
def set_clear_color(red, green, blue, alpha):
	pass

def set_dump_format(f):
	global _dump_format
	if _dump_format == f:
		return
	_dump_format = f
	if _dump_format == JSON_FORMAT:
		DataType.reinit()
		ShaderType.reinit()
		BufferTarget.reinit()
	else:
		DataType.reinit(True)
		ShaderType.reinit(True)
		BufferTarget.reinit(True)
