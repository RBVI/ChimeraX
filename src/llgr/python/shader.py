from OpenGL import GL
from OpenGL.GL import shaders
import struct

_current_program = None

def check_GLerror(message):
	error = GL.glGetError()
	if error == GL.GL_NO_ERROR:
		return
	raise RuntimeError("%s (%s)" % (message, hex(error)))

class ShaderVariable:
	# cached introspected information about shader program
	# uniforms and attributes

	(Unknown, Float, Vec2, Vec3, Vec4,
	Int, IVec2, IVec3, IVec4,
	UInt, UVec2, UVec3, UVec4,
	Bool, BVec2, BVec3, BVec4,
	Mat2x2, Mat3x3, Mat4x4,
	Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3) = range(26)

	def __init__(self, shader_program, name, type, location):
		self.sp = shader_program
		self.name = name
		self.type = type
		self.location = location
		self.data = None
		self.transpose = False

	def has_value(self):
		return self.data is not None

	_count_map = {
		Float: 1, Vec2: 2, Vec3: 3, Vec4: 4,
		Int: 1, IVec2: 2, IVec3: 3, IVec4: 4,
		UInt: 1, UVec2: 2, UVec3: 3, UVec4: 4,
		Bool: 1, BVec2: 2, BVec3: 3, BVec4: 4,
		Mat2x2: 4, Mat3x3: 9, Mat4x4: 16,
		Mat2x3: 6, Mat3x2: 6,
		Mat2x4: 8, Mat4x2: 8,
		Mat3x4: 12, Mat4x3: 12,
	}

	def count(self):
		return self._count_map.get(self.type, 0)

	_base_type_map = {
		Float: Float, Vec2: Float, Vec3: Float, Vec4: Float,
		Int: Int, IVec2: Int, IVec3: Int, IVec4: Int,
		UInt: UInt, UVec2: UInt, UVec3: UInt, UVec4: UInt,
		Bool: Bool, BVec2: Bool, BVec3: Bool, BVec4: Bool,
		Mat2x2: Float, Mat3x3: Float, Mat4x4: Float,
		Mat2x3: Float, Mat3x2: Float,
		Mat2x4: Float, Mat4x2: Float,
		Mat3x4: Float, Mat4x3: Float,
	}

	def base_type(self):
		return self._base_type_map.get(self.type, self.Unknown)

	_location_info_map = {
		# (number of locations used, number of elements per location)
		Float: (1, 1), Vec2: (1, 2), Vec3: (1, 3), Vec4: (1, 4),
		Int: (1, 1), IVec2: (1, 2), IVec3: (1, 3), IVec4: (1, 4),
		UInt: (1, 1), UVec2: (1, 2), UVec3: (1, 3), UVec4: (1, 4),
		Bool: (1, 1), BVec2: (1, 2), BVec3: (1, 3), BVec4: (1, 4),
		Mat2x2: (2, 2), Mat3x3: (3, 3), Mat4x4: (4, 4),
		Mat2x3: (2, 3), Mat3x2: (3, 2),
		Mat2x4: (2, 4), Mat4x2: (4, 2),
		Mat3x4: (3, 4), Mat4x3: (4, 3),
	}

	def location_info(self):
		return self._location_info_map.get(self.type, (0, 0))

	@staticmethod
	def type_location_info(type):
		return ShaderVariable._location_info_map.get(type, (0, 0))

	def byte_count(self):
		# all type are represented by 4 byte values for draw_uniform
		return self.count() * 4

	def set_float(self, f):
		if self.type != self.Float:
			raise ValueError("not a float point singleton")
		self.data = struct.pack("@f", f)
		if len(self.data) != self.byte_count():
			raise ValueError("expected %d bytes, got %d" % (self.byte_count(), len(self.data)))
		if _current_program == self.sp:
			self.draw_uniform()

	def set_floatv(self, fv):
		if self.base_type() != self.Float:
			raise ValueError("not a float point type")
		self.data = fv
		if _current_program == self.sp:
			self.draw_uniform()

	def set_float_matrixv(self, transpose, fv):
		if self.base_type() != self.Float:
			raise ValueError("not a float point type")
		try:
			self.data = fv
		except TypeError:
			self.data = struct.pack("@f" * self.count(), *fv)
		self.transpose = transpose
		if _current_program == self.sp:
			self.draw_uniform()

	def set_int(self, i):
		if self.type != self.Int:
			raise ValueError("not an integer singleton")
		self.data = struct.pack("@i", i)
		if len(self.data) != self.byte_count():
			raise ValueError("expected %d bytes, got %d" % (self.byte_count(), len(self.data)))
		if _current_program == self.sp:
			self.draw_uniform()

	def set_intv(self, iv):
		if self.base_type() not in (self.Int, self.UInt):
			raise ValueError("not an integer type")
		self.data = iv
		if _current_program == self.sp:
			self.draw_uniform()

	def set_bool(self, b):
		if self.type != self.Bool:
			raise ValueError("not a boolean singleton")
		self.data = struct.pack("@i", b)
		if len(self.data) != self.byte_count():
			raise ValueError("expected %d bytes, got %d" % (self.byte_count(), len(self.data)))
		if _current_program == self.sp:
			self.draw_uniform()

	def set_boolv(self, bv):
		if self.base_type() != self.Bool:
			raise ValueError("not a boolean type")
		self.data = bytes(bv)
		if len(self.data) != self.byte_count():
			raise ValueError("expected %d bytes, got %d" % (self.byte_count(), len(self.data)))
		if _current_program == self.sp:
			self.draw_uniform()

	_draw_uniform_map = {
		Float: GL.glUniform1fv,
		Vec2: GL.glUniform2fv,
		Vec3: GL.glUniform3fv,
		Vec4: GL.glUniform4fv,

		Int: GL.glUniform1iv,
		IVec2: GL.glUniform2iv,
		IVec3: GL.glUniform3iv,
		IVec4: GL.glUniform4iv,

		UInt: GL.glUniform1uiv,
		UVec2: GL.glUniform2uiv,
		UVec3: GL.glUniform3uiv,
		UVec4: GL.glUniform4ui,

		Bool: GL.glUniform1iv,
		BVec2: GL.glUniform2iv,
		BVec3: GL.glUniform3iv,
		BVec4: GL.glUniform4iv,

		Mat2x2: GL.glUniformMatrix2fv,
		Mat3x3: GL.glUniformMatrix3fv,
		Mat4x4: GL.glUniformMatrix4fv,

		#Mat2x3: shaders.glUniform2x3fv,
		#Mat3x2: shaders.glUniform3x2fv,
		#Mat2x4: shaders.glUniform2x4fv,
		#Mat4x2: shaders.glUniform4x2fv,
		#Mat3x4: shaders.glUniform3x4fv,
		#Mat4x3: shaders.glUniform4x3fv,
	}
	def draw_uniform(self):
		if self.data is None:
			return
		func = self._draw_uniform_map[self.type]
		if self.type >= self.Mat2x2:
			func(self.location, 1, self.transpose, self.data)
		else:
			func(self.location, 1, self.data)

_type_map = {
	GL.GL_FLOAT:		ShaderVariable.Float,
	GL.GL_FLOAT_VEC2:	ShaderVariable.Vec2,
	GL.GL_FLOAT_VEC3:	ShaderVariable.Vec3,
	GL.GL_FLOAT_VEC4:	ShaderVariable.Vec4,
	GL.GL_INT:		ShaderVariable.Int,
	GL.GL_INT_VEC2:		ShaderVariable.IVec2,
	GL.GL_INT_VEC3:		ShaderVariable.IVec3,
	GL.GL_INT_VEC4:		ShaderVariable.IVec4,
	GL.GL_UNSIGNED_INT:	ShaderVariable.UInt,
	GL.GL_UNSIGNED_INT_VEC2:	ShaderVariable.UVec2,
	GL.GL_UNSIGNED_INT_VEC3:	ShaderVariable.UVec3,
	GL.GL_UNSIGNED_INT_VEC4:	ShaderVariable.UVec4,
	GL.GL_BOOL:		ShaderVariable.Bool,
	GL.GL_BOOL_VEC2:	ShaderVariable.BVec2,
	GL.GL_BOOL_VEC3:	ShaderVariable.BVec3,
	GL.GL_BOOL_VEC4:	ShaderVariable.BVec4,
	GL.GL_FLOAT_MAT2:	ShaderVariable.Mat2x2,
	GL.GL_FLOAT_MAT3:	ShaderVariable.Mat3x3,
	GL.GL_FLOAT_MAT4:	ShaderVariable.Mat4x4,
	GL.GL_FLOAT_MAT2x3:	ShaderVariable.Mat2x3,
	GL.GL_FLOAT_MAT3x2:	ShaderVariable.Mat3x2,
	GL.GL_FLOAT_MAT2x4:	ShaderVariable.Mat2x4,
	GL.GL_FLOAT_MAT4x2:	ShaderVariable.Mat4x2,
	GL.GL_FLOAT_MAT3x4:	ShaderVariable.Mat3x4,
	GL.GL_FLOAT_MAT4x3:	ShaderVariable.Mat4x3,
#ifdef HAVE_TEXTURE
#	GL.GL_SAMPLER_1D:	ShaderVariable.Sampler1D,
#	GL.GL_SAMPLER_2D:	ShaderVariable.Sampler2D,
#	GL.GL_SAMPLER_3D:	ShaderVariable.Sampler3D,
#	GL.GL_SAMPLER_CUBE:	ShaderVariable.SamplerCube,
#	GL.GL_SAMPLER_1D_SHADOW:	ShaderVariable.Sampler1DShadow,
#	GL.GL_SAMPLER_2D_SHADOW:	ShaderVariable.Sampler2DShadow,
#endif
}

def cvt_type(type: GL.GLenum):
	return _type_map.get(type, ShaderVariable.Unknown)

class ShaderProgram:

	def __init__(self, vertex_shader, fragment_shader, attribute0_name):
		import sys
		self.program = shaders.glCreateProgram()
		if self.program == 0:
			check_GLerror()
			return
		self.uniforms = []
		self.attributes = []

		self.vs = shaders.glCreateShader(GL.GL_VERTEX_SHADER)
		shaders.glShaderSource(self.vs, vertex_shader)
		shaders.glCompileShader(self.vs)
		shaders.glAttachShader(self.program, self.vs)

		self.fs = shaders.glCreateShader(GL.GL_FRAGMENT_SHADER)
		shaders.glShaderSource(self.fs, fragment_shader)
		shaders.glCompileShader(self.fs)
		shaders.glAttachShader(self.program, self.fs)

		compiled = True
		status = shaders.glGetShaderiv(self.vs, GL.GL_COMPILE_STATUS)
		if not status:
			compiled = False
			log = shaders.gletShaderLog(self.vs)
			print("compiling vertex shader failed:\n%s" % log, file=sys.stderr)
		status = shaders.glGetShaderiv(self.fs, GL.GL_COMPILE_STATUS)
		if not status:
			compiled = False
			log = shaders.gletShaderLog(self.fs)
			print("compiling fragment shader failed:\n%s" % log, file=sys.stderr)
		if not compiled:
			raise RuntimeError("failed to compile shader program")

		if attribute0_name:
			shaders.glBindAttribLocation(self.program, 0, attribute0_name.encode('utf-8'))

		shaders.glLinkProgram(self.program)
		status = shaders.glGetProgramiv(self.program, GL.GL_LINK_STATUS)
		if not status:
			log = shaders.gletProgramInfoLog(self.program)
			print("unable to link program:\n%s" % log, file=sys.stderr)

		# introspect program uniforms
		num_uniforms = shaders.glGetProgramiv(self.program, GL.GL_ACTIVE_UNIFORMS)
		for i in range(num_uniforms):
			name, size, type = shaders.glGetActiveUniform(self.program, i)
			if size != 1:
				print("uniform arrays are not supported", file=sys.stderr)
			loc = shaders.glGetUniformLocation(self.program, name)
			if loc == -1:
				continue
			self.uniforms.append(ShaderVariable(self, name.decode('utf-8'), cvt_type(type), loc))

		# introspect vertex attributes
		num_attributes = shaders.glGetProgramiv(self.program, GL.GL_ACTIVE_ATTRIBUTES)
		for i in range(num_attributes):
			name, size, type = shaders.glGetActiveAttrib(self.program, i)
			if size != 1:
				print("attribue arrays are not supported", file=sys.stderr)
			loc = shaders.glGetAttribLocation(self.program, name)
			if loc == -1:
				continue
			self.attributes.append(ShaderVariable(self, name.decode('utf-8'), cvt_type(type), loc))

	def close(self):
		if self.program:
			shaders.glDeleteProgram(self.program)
		if self.vs:
			shaders.glDeleteShader(self.vs)
		if self.fs:
			shaders.glDeleteShader(self.fs)

	def uniform(self, name, exceptions=False):
		for u in self.uniforms:
			if u.name == name:
				return u
		if exceptions:
			raise ValueError("uniform not found: %s" % name)
		return None

	def attribute(self, name, exceptions=False):
		for a in self.attributes:
			if a.name == name:
				return a
		if exceptions:
			raise ValueError("attribute not found: %s" % name)
		return None

	def setup(self):
		if not self.program:
			return

		shaders.glUseProgram(self.program)
		global _current_program
		_current_program = self.program
		for u in self.uniforms:
			u.draw_uniform()

	def cleanup(self):
		global _current_program
		if not self.program:
			return
		GL.glUseProgram(0)
		_current_program = 0
