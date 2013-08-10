"""
Dump llgr calls.  Emit when render() is called.

The basic llgr interface is extended to manage Ids.

Add typechecking to function arguments to protect against arguments with the
wrong type being decoded by matching backend in JavaScript.
"""
import numpy
from typecheck import typecheck, Checker, either, list_of, TypeCheckError

JSON_FORMAT = 'json'
CPP_FORMAT = 'c++'
JS_FORMAT = 'js'

FORMATS = (JSON_FORMAT, CPP_FORMAT, JS_FORMAT)

_dump_format = JSON_FORMAT

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

	def check(self, value):
		if _dump_format == JSON_FORMAT:
			if value not in self.values:
				return False
		if value not in self.labels:
			return False
		return True

class DataType(Enum):
	values = list(range(7))
	labels = ('Byte', 'UByte', 'Short', 'UShort', 'Int', 'UInt', 'Float')
DataType.class_init()

class ShaderType(Enum):
	values = list(range(21))
	labels = ('IVec1', 'IVec2', 'IVec3', 'IVec4', 'UVec1', 'UVec2', 'UVec3', 'UVec4', 'FVec1', 'FVec2', 'FVec3', 'FVec4', 'Mat2x2', 'Mat3x3', 'Mat4x4',
	'Mat2x3', 'Mat3x2', 'Mat2x4',
	'Mat4x2', 'Mat3x4', 'Mat4x3')
ShaderType.class_init()

def convert_buffer(b):
	"""return [little-endian, size, [unsigned integers]]"""
	b = memoryview(b)
	if _dump_format != JSON_FORMAT:
		ctype = "missing"
		if _dump_format == CPP_FORMAT:
			ctype = "const char*"
		elif _dump_format == JS_FORMAT:
			ctype = "var"
		text = repr(bytes(b))[1:-1].replace('"', '\\"')
		return ctype, '"%s"' % text, len(b)
	import struct
	fmt = 'I' * (b.nbytes // 4)
	fmt = '<' + fmt + ['', 'B', 'H', 'HB'][b.nbytes % 4]
	return [True, b.nbytes, struct.unpack(fmt, b)]

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

@typecheck
def render():
	global _calls
	tmp = _calls
	_calls = []
	if _dump_format == JSON_FORMAT:
		import json
		from json.encoder import JSONEncoder
		class DumpEncoder(JSONEncoder):
			def default(self, obj):
				if isinstance(obj, Enum):
					return obj.value
				if isinstance(obj, AttributeInfo):
					return obj.json()
				return JSONEncoder.default(self, obj)
		with open("render.json", "w") as f:
			json.dump(tmp, f, cls=DumpEncoder)
			print(file=f)	# trailing newline
		return
	if _dump_format == CPP_FORMAT:
		with open("render.cpp", "w") as f:
			print("#include <llgr.h>\n", file=f)
			print("#define False 0\n", file=f)
			print("#define True 1\n", file=f)
			print("using namespace llgr;\n", file=f)
			print("void\ninit_objects()\n{", file=f)
			for call in tmp:
				print('%s' % call, file=f)
			print("}", file=f)
		return
	if _dump_format == JS_FORMAT:
		with open("render.js", "w") as f:
			print("function init_objects()\n{", file=f)
			for call in tmp:
				print("%s;" % call, file=f)
			print("}", file=f)

@typecheck
def create_program(program_id: Id, vertex_shader: str, fragment_shader: str, pick_vertex_shader: str):
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
@typecheck
def delete_program(program_id: Id):
	pass

@save_args
@typecheck
def clear_programs():
	pass

@typecheck
def set_uniform(program_id: Id, name: str, shader_type: ShaderType, data: IsBuffer):
	if _dump_format == JSON_FORMAT:
		_calls.append(['set_uniform', [program_id, name, shader_type, convert_buffer(data)]])
		return
	uname = 'u_%s%s' % (name, program_id)
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, uname, init))
	_calls.append('\tset_uniform(%s, "%s", %s, %s, %s);'
			% (program_id, name, shader_type, num_bytes, uname))

@typecheck
def set_uniform_matrix(program_id: Id, name: str, transpose: bool, shader_type: ShaderType, data: IsBuffer):
	if _dump_format == JSON_FORMAT:
		_calls.append(['set_uniform_matrix', [program_id, name, transpose, shader_type, convert_buffer(data)]])
		return
	uname = 'u_%s%s' % (name, program_id)
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, uname, init))
	_calls.append('\tset_uniform_matrix(%s, "%s", %s, %s, %s, %s);'
		% (program_id, name, transpose, shader_type, num_bytes, uname))

# (interleaved) buffer support

class BufferTarget(Enum):
	values = (0x8892, 0x8893)
	labels = ('ARRAY', 'ELEMENT_ARRAY')
BufferTarget.class_init()

# Create buffer of array data
@typecheck
def create_buffer(data_id: Id, buffer_target: BufferTarget, data: IsBuffer):
	if _dump_format == JSON_FORMAT:
		_calls.append(['create_buffer', [data_id, buffer_target, convert_buffer(data)]])
		return
	dname = ('d%s' % data_id).replace('-', '_')
	type, init, num_bytes = convert_buffer(data)
	_calls.append('\t%s %s = %s;' % (type, dname, init))
	_calls.append('\tcreate_buffer(%s, %s, %s, %s);'
				% (data_id, buffer_target, num_bytes, dname))

@save_args
@typecheck
def delete_buffer(data_id: Id):
	pass

@save_args
@typecheck
def clear_buffers():
	pass

# create singleton "buffer" data
@typecheck
def create_singleton(data_id: Id, data: IsBuffer):
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
@typecheck
def create_singleton_index(data_id: Id, reference_data_id: Id, size: NonNeg32, offset: NonNeg32):
	pass # TODO

# update column of existing buffer data
@typecheck
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

@typecheck
def create_2d_texture(tex_id: Id, texture_format: TextureFormat, texture_min_filter: TextureFilter, texture_max_filter: TextureFilter, data_type: DataType, width: NonNeg32, height: NonNeg32, data: IsBuffer):
	pass # TODO

@typecheck
def create_3d_texture(tex_id: Id, texture_format: TextureFormat, texture_min_filter: TextureFilter, texture_max_filter: TextureFilter, data_type: DataType, width: NonNeg32, height: NonNeg32, depth: NonNeg32, data: IsBuffer):
	pass # TODO

@save_args
@typecheck
def delete_texture(data_id: Id):
	pass # TODO

@save_args
@typecheck
def clear_textures():
	pass

# matrices

# type check for 4x4 array, dtype='f' (aka dtype('float32'))
Matrix_4x4 = either(Array(shape=(4,4), dtype='f'), (
	float, float, float, float,
	float, float, float, float,
	float, float, float, float,
	float, float, float, float
))

# matrix_id of zero is reserved for identity matrix
# renormalize should be true when the rotation part of the matrix
# has shear or scaling, or if it is a projection matrix. 
@typecheck
def create_matrix(matrix_id: Id, matrix_4x4: Matrix_4x4, renormalize: bool=False):
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
@typecheck
def delete_matrix(matrix_id: Id):
	pass

@save_args
@typecheck
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

class AttributeInfo(object):

	@typecheck
	def __init__(self, name: str, data_id: Id, offset: NonNeg32, stride: NonNeg32, count: NonNeg32, data_type: DataType, norm: bool=False):
		self.name = name
		self.data_id = data_id
		self.offset = offset
		self.stride = stride
		self.count = count
		self.data_type = data_type
		self.normalize = norm

	def __repr__(self):
		return 'AttributeInfo("%s", %r, %r, %r, %r, %r, %r)' % (
			self.name, self.data_id, self.offset, self.stride,
			self.count, self.data_type, self.normalize)

	def json(self):
		return (self.name, self.data_id, self.offset, self.stride, self.count, self.data_type, self.normalize)

# enum PrimitiveType
class PrimitiveType(Enum):
	values = list(range(7))
	labels = ('Points', 'Lines', 'Line_loop', 'Line_strip',
			'Triangles', 'Triangle_strip', 'Triangle_fan')
PrimitiveType.class_init()

@save_args
@typecheck
def set_attribute_alias(name: str, value: str):
	pass

@save_args
@typecheck
def create_object(obj_id: Id, program_id: Id, matrix_id: Id,
		list_of_attributeInfo: list_of(AttributeInfo),
		primitive_type: PrimitiveType, first: NonNeg32, count: NonNeg32,
		index_data_id: Id=0, index_buffer_type: DataType=UByte):
	pass

@save_args
@typecheck
def delete_object(obj_id: Id):
	pass

@save_args
@typecheck
def clear_objects():
	pass

# indicate whether to draw object or not
@save_args
@typecheck
def hide_objects(list_of_objects: list_of(Id)):
	pass

@save_args
@typecheck
def show_objects(list_of_objects: list_of(Id)):
	pass

# indicate whether an object is transparent or opaque (default opaque)
@save_args
@typecheck
def transparent(list_of_objects: list_of(Id)):
	pass

@save_args
@typecheck
def opaque(list_of_objects: list_of(Id)):
	pass

@save_args
@typecheck
def selection_add(list_of_objects: list_of(Id)):
	pass

@save_args
@typecheck
def selection_remove(list_of_objects: list_of(Id)):
	pass

@save_args
@typecheck
def selection_clear():
	pass

# TODO: text primitives

# LOD primitives

@typecheck
def add_sphere(obj_id: Id, radius: Number, program_id: Id, matrix_id: Id, list_of_attributeInfo: list_of(AttributeInfo)):
	if _dump_format == JSON_FORMAT:
		_calls.append(['add_sphere', [obj_id, radius, program_id, matrix_id, list_of_attributeInfo]])
		return
	aname = ('a%s' % obj_id).replace('-', '_')
	_calls.append('\tAttributeInfos %s;' % aname)
	for ai in list_of_attributeInfo:
		_calls.append("\t%s.push_back(%s);" % (aname, ai))
	_calls.append('\tadd_sphere(%r, %r, %r, %r, %s, "%s", "%s");'
			% (obj_id, radius, program_id, matrix_id, aname))

@typecheck
def add_cylinder(obj_id: Id, radius: Number, length: Number,
		program_id: Id, matrix_id: Id, list_of_attributeInfo: list_of(AttributeInfo)):
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
@typecheck
def clear_primitives():
	pass

@save_args
@typecheck
def clear_all():
	pass

@save_args
@typecheck
def set_clear_color(red: GLclampf, green: GLclampf, blue: GLclampf, alpha: GLclampf):
	pass

def set_dump_format(f):
	global _dump_format
	if _dump_format == f:
		return
	_dump_format = f
	if _dump_format == JSON_FORMAT:
		DataType.class_init()
		ShaderType.class_init()
		BufferTarget.class_init()
	else:
		DataType.class_init(True)
		ShaderType.class_init(True)
		BufferTarget.class_init(True)
