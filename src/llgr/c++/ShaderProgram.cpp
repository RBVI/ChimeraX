//#define GL3_PROTOTYPES
//#define GL_GLEXT_PROTOTYPES
#include "ShaderProgram.h"
#define GLEW_NO_GLU
#include <GL/glew.h>
#include <typeinfo>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <locale>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits.h>
#include <string.h>
#include <assert.h>

#undef PRINT_SHADER_INFO

namespace llgr {

using std::string;

#ifndef OTF_NO_LOCALE
const std::ctype<string::value_type> &ct
    = std::use_facet<std::ctype<string::value_type>>(std::locale::classic());
#endif

GLuint	current_programObj;

bool
hasGLError(const char *message)
{
	GLenum error = glGetError();
	if (error == GL_NO_ERROR)
		return false;
	std::cerr << message << ": '" << glewGetErrorString(error) << "' (0x"
				<< std::hex << error << std::dec << ")\n";
	return true;
}

ShaderVariable::Type
cvt_type(GLenum type)
{
	switch (type) {
	  case GL_FLOAT:	return ShaderVariable::Float;
	  case GL_FLOAT_VEC2:	return ShaderVariable::Vec2;
	  case GL_FLOAT_VEC3:	return ShaderVariable::Vec3;
	  case GL_FLOAT_VEC4:	return ShaderVariable::Vec4;
	  case GL_INT:		return ShaderVariable::Int;
	  case GL_INT_VEC2:	return ShaderVariable::IVec2;
	  case GL_INT_VEC3:	return ShaderVariable::IVec3;
	  case GL_INT_VEC4:	return ShaderVariable::IVec4;
	  case GL_UNSIGNED_INT:	return ShaderVariable::UInt;
	  case GL_UNSIGNED_INT_VEC2:	return ShaderVariable::UVec2;
	  case GL_UNSIGNED_INT_VEC3:	return ShaderVariable::UVec3;
	  case GL_UNSIGNED_INT_VEC4:	return ShaderVariable::UVec4;
	  case GL_BOOL:		return ShaderVariable::Bool;
	  case GL_BOOL_VEC2:	return ShaderVariable::BVec2;
	  case GL_BOOL_VEC3:	return ShaderVariable::BVec3;
	  case GL_BOOL_VEC4:	return ShaderVariable::BVec4;
	  case GL_FLOAT_MAT2:	return ShaderVariable::Mat2x2;
	  case GL_FLOAT_MAT3:	return ShaderVariable::Mat3x3;
	  case GL_FLOAT_MAT4:	return ShaderVariable::Mat4x4;
	  case GL_FLOAT_MAT2x3:	return ShaderVariable::Mat2x3;
	  case GL_FLOAT_MAT3x2:	return ShaderVariable::Mat3x2;
	  case GL_FLOAT_MAT2x4:	return ShaderVariable::Mat2x4;
	  case GL_FLOAT_MAT4x2:	return ShaderVariable::Mat4x2;
	  case GL_FLOAT_MAT3x4:	return ShaderVariable::Mat3x4;
	  case GL_FLOAT_MAT4x3:	return ShaderVariable::Mat4x3;
#ifdef HAVE_TEXTURE
	  case GL_SAMPLER_1D:	return ShaderVariable::Sampler1D;
	  case GL_SAMPLER_2D:	return ShaderVariable::Sampler2D;
	  case GL_SAMPLER_3D:	return ShaderVariable::Sampler3D;
	  case GL_SAMPLER_CUBE:	return ShaderVariable::SamplerCube;
	  case GL_SAMPLER_1D_SHADOW:	return ShaderVariable::Sampler1DShadow;
	  case GL_SAMPLER_2D_SHADOW:	return ShaderVariable::Sampler2DShadow;
#endif
	}
	return ShaderVariable::Unknown;
}


ShaderVariable::ShaderVariable(ShaderProgram *sp, const string& n, Type t):
	sp_(sp), name_(n), transpose_(false), type_(t), location_(-1), data(NULL)
{
}

ShaderVariable::~ShaderVariable()
{
	delete [] data;
}

unsigned
ShaderVariable::count() const
{
	switch (type_) {
	  case Float: return 1;
	  case Vec2: return 2;
	  case Vec3: return 3;
	  case Vec4: return 4;

	  case Int: return 1;
	  case IVec2: return 2;
	  case IVec3: return 3;
	  case IVec4: return 4;

	  case UInt: return 1;
	  case UVec2: return 2;
	  case UVec3: return 3;
	  case UVec4: return 4;

	  case Bool: return 1;
	  case BVec2: return 2;
	  case BVec3: return 3;
	  case BVec4: return 4;

	  case Mat2x2: return 4;
	  case Mat3x3: return 9;
	  case Mat4x4: return 16;

	  case Mat2x3: return 6;
	  case Mat3x2: return 6;
	  case Mat2x4: return 8;
	  case Mat4x2: return 8;
	  case Mat3x4: return 12;
	  case Mat4x3: return 12;

#ifdef HAVE_TEXTURE
	  case Sampler1D: return ?;
	  case Sampler2D: return ?;
	  case Sampler3D: return ?;
	  case SamplerCube: return ?;
	  case Sampler1DShadow: return ?;
	  case Sampler2DShadow: return ?;
#endif
	  default: return 0;
	}
}

ShaderVariable::Type
ShaderVariable::base_type() const
{
	switch (type_) {
	  case Float: case Vec2: case Vec3: case Vec4:
		  return Float;

	  case Int: case IVec2: case IVec3: case IVec4:
		  return Int;

	  case UInt: case UVec2: case UVec3: case UVec4:
		  return UInt;

	  case Bool: case BVec2: case BVec3: case BVec4:
		  return Bool;

	  case Mat2x2: case Mat3x3: case Mat4x4:
	  case Mat2x3: case Mat3x2:
	  case Mat2x4: case Mat4x2:
	  case Mat3x4: case Mat4x3:
		  return Float;

#ifdef HAVE_TEXTURE
	  case Sampler1D: return Sampler1D;
	  case Sampler2D: return Sampler2D;
	  case Sampler3D: return Sampler3D;
	  case SamplerCube: return Sampler2D;
	  case Sampler1DShadow: return Sampler1D;
	  case Sampler2DShadow: return Sampler2D;
#endif
	  default: return Unknown;
	}
}

unsigned
ShaderVariable::byte_count() const
{
	switch (type_) {
	  case Float: case Vec2: case Vec3: case Vec4:
		  return count() * sizeof (GLfloat);

	  case Int: case IVec2: case IVec3: case IVec4:
		  return count() * sizeof (GLint);

	  case UInt: case UVec2: case UVec3: case UVec4:
		  return count() * sizeof (GLuint);

	  case Bool: case BVec2: case BVec3: case BVec4:
		  return count() * sizeof (GLint);

	  case Mat2x2: case Mat3x3: case Mat4x4:
	  case Mat2x3: case Mat3x2:
	  case Mat2x4: case Mat4x2:
	  case Mat3x4: case Mat4x3:
		  return count() * sizeof (GLfloat);

#ifdef HAVE_TEXTURE
	  case Sampler1D: return ?;
	  case Sampler2D: return ?;
	  case Sampler3D: return ?;
	  case SamplerCube: return ?;
	  case Sampler1DShadow: return ?;
	  case Sampler2DShadow: return ?;
#endif
	  default: return 0;
	}
}

void
ShaderVariable::set_float(GLfloat f)
{
	if (type_ != Float)
		throw std::logic_error("not a floating point singleton");
	if (!data)
		data = new unsigned char [sizeof (GLfloat)];
	*reinterpret_cast<GLfloat *>(data) = f;
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

void
ShaderVariable::set_floatv(const GLfloat *fv)
{
	if (fv == NULL)
		throw std::logic_error("need float array");
	if (base_type() != Float)
		throw std::logic_error("not a floating point type");
	unsigned num_bytes = byte_count();
	if (!data)
		data = new unsigned char [num_bytes];
	memcpy(data, fv, num_bytes);
	transpose_ = false;
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

void
ShaderVariable::set_float_matrixv(bool transpose, const GLfloat *fv)
{
	if (fv == NULL)
		throw std::logic_error("need float array");
	if (base_type() != Float)
		throw std::logic_error("not a floating point type");
	unsigned num_bytes = byte_count();
	if (!data)
		data = new unsigned char [num_bytes];
	memcpy(data, fv, num_bytes);
	transpose_ = transpose;
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

void
ShaderVariable::set_int(GLint i)
{
	if (type_ != Int)
		throw std::logic_error("not an integer singleton");
	if (!data)
		data = new unsigned char [sizeof (GLint)];
	*reinterpret_cast<GLint *>(data) = i;
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

void
ShaderVariable::set_intv(const int *iv)
{
	if (iv == NULL)
		throw std::logic_error("need integer array");
	if (base_type() != Int)
		throw std::logic_error("not an integer type");
	unsigned num_bytes = byte_count();
	if (!data)
		data = new unsigned char [num_bytes];
	memcpy(data, iv, num_bytes);
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

void
ShaderVariable::set_bool(GLint b)
{
	if (type_ != Bool)
		throw std::logic_error("not a boolean singleton");
	if (!data)
		data = new unsigned char [sizeof (GLint)];
	*reinterpret_cast<GLint *>(data) = b;
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

void
ShaderVariable::set_boolv(const GLint *bv)
{
	if (bv == NULL)
		throw std::logic_error("need boolean array");
	if (base_type() != Bool)
		throw std::logic_error("not a boolean type");
	unsigned num_bytes = byte_count();
	if (!data)
		data = new unsigned char [num_bytes];
	memcpy(data, bv, num_bytes);
	if (current_programObj == sp_->programObj)
		draw_uniform();
}

#ifdef HAVE_TEXTURE
Texture*
ShaderVariable::texture() const
{
	if (!has_value())
		throw std::logic_error("no value set");
	switch (type_) {
	  default:
		throw std::logic_error("not a sampler type");
	  case Sampler1D: case Sampler2D: case Sampler3D: case SamplerCube:
	  case Sampler1DShadow: case Sampler2DShadow:
	  	return data.tex;
	} }

void
ShaderVariable::set_texture(Texture *t)
{
	if (t == NULL)
		throw std::logic_error("need a texture");
	switch (type_) {
	  default:
		throw std::logic_error("not a sampler type");
	  case Sampler1D: case Sampler1DShadow:
		if (t->dimension() != 1)
			throw std::logic_error("not a 1D texture");
		break;
	  case Sampler2D: case SamplerCube: case Sampler2DShadow:
		if (t->dimension() != 2)
			throw std::logic_error("not a 2D texture");
		break;
	  case Sampler3D:
		if (t->dimension() != 3)
			throw std::logic_error("not a 3D texture");
		break;
	}
	data.tex = t;
	if (current_programObj == sp_->programObj)
		draw_uniform();
}
#endif

void
ShaderVariable::draw_uniform() const
{
	switch (type_) {
	  case Float: glUniform1fv(location_, 1, reinterpret_cast<GLfloat *>(data)); break;
	  case Vec2: glUniform2fv(location_, 1, reinterpret_cast<GLfloat *>(data)); break;
	  case Vec3: glUniform3fv(location_, 1, reinterpret_cast<GLfloat *>(data)); break;
	  case Vec4: glUniform4fv(location_, 1, reinterpret_cast<GLfloat *>(data)); break;

	  case Int: glUniform1iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case IVec2: glUniform2iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case IVec3: glUniform3iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case IVec4: glUniform4iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case UInt: glUniform1uiv(location_, 1, reinterpret_cast<GLuint *>(data)); break;
	  case UVec2: glUniform2uiv(location_, 1, reinterpret_cast<GLuint *>(data)); break;
	  case UVec3: glUniform3uiv(location_, 1, reinterpret_cast<GLuint *>(data)); break;
	  case UVec4: glUniform4uiv(location_, 1, reinterpret_cast<GLuint *>(data)); break;

	  case Bool: glUniform1iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case BVec2: glUniform2iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case BVec3: glUniform3iv(location_, 1, reinterpret_cast<GLint *>(data)); break;
	  case BVec4: glUniform4iv(location_, 1, reinterpret_cast<GLint *>(data)); break;

	  case Mat2x2: glUniformMatrix2fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat3x3: glUniformMatrix3fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat4x4: glUniformMatrix4fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;

	  case Mat2x3: glUniformMatrix2x3fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat3x2: glUniformMatrix3x2fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat2x4: glUniformMatrix2x4fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat4x2: glUniformMatrix4x2fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat3x4: glUniformMatrix3x4fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;
	  case Mat4x3: glUniformMatrix4x3fv(location_, 1, transpose_, reinterpret_cast<GLfloat *>(data)); break;

#ifdef HAVE_TEXTURE
	  case Sampler1D:
	  case Sampler2D:
	  case Sampler3D:
	  case SamplerCube:
	  case Sampler1DShadow:
	  case Sampler2DShadow:
		throw std::runtime_error("not implemented yet");
	  	// glUniform1iv(location_, 1, &textureUnit); 
	  	break;
#endif
	  case Unknown:
	  	break;
	}
}

inline bool
isSpace(string::traits_type::char_type c)
{
#ifdef OTF_NO_LOCALE
	return isspace(c);
#else
	return ct.is(ct.space, c);
#endif
}

ShaderProgram::ShaderProgram(const string& vertex_shader, const string& fragment_shader, const string& attribute0_name): programObj(0), vs(0), fs(0)
{
	class GuardProgram {
		// Protect against leaks due to exceptions
		GLuint program;
	public:
		GuardProgram(): program(glCreateProgram())
			{
				if (program == 0) {
					GLenum err = glGetError();
					string msg("unable to create program object: ");
					msg += (char *) glewGetErrorString(err);
					throw std::runtime_error(msg);
				}
			}
		~GuardProgram() {
				if (program)
					glDeleteProgram(program);
			}
		GLuint get() {
				return program;
			}
		GLuint release() {
				GLuint tmp = program;
				program = 0;
				return tmp;
			}
	};

	GuardProgram program;

	const GLchar *sourcev[1];
	vs = glCreateShader(GL_VERTEX_SHADER);
	sourcev[0] = reinterpret_cast<const GLchar *>(vertex_shader.c_str());
	glShaderSource(vs, 1, sourcev, NULL);
	glCompileShader(vs);
	glAttachShader(program.get(), vs);

	fs = glCreateShader(GL_FRAGMENT_SHADER);
	sourcev[0] = reinterpret_cast<const GLchar *>(fragment_shader.c_str());
	glShaderSource(fs, 1, sourcev, NULL);
	glCompileShader(fs);
	glAttachShader(program.get(), fs);

	// check that all shaders compiled and linked
	bool compiled = true;
	GLint status;
	glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE) {
		compiled = false;
		GLsizei length;
		glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &length);
		std::vector<GLchar> msg(length + 1);
		glGetShaderInfoLog(vs, length, 0, &msg[0]);
		std::cerr << "compiling vertex shader failed.\n";
		std::cerr << (&msg[0]) << '\n';
	}
	glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE) {
		compiled = false;
		GLsizei length;
		glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &length);
		std::vector<GLchar> msg(length + 1);
		glGetShaderInfoLog(fs, length, 0, &msg[0]);
		std::cerr << "compiling vertex shader failed.\n";
		std::cerr << (&msg[0]) << '\n';
	}
	if (!compiled)
		throw std::runtime_error("unable to compile all shaders");

	if (!attribute0_name.empty()) {
		glBindAttribLocation(program.get(), 0, attribute0_name.c_str());
	}

	glLinkProgram(program.get());
	glGetProgramiv(program.get(), GL_LINK_STATUS, &status);
	if (status != GL_TRUE) {
		GLsizei length;
		glGetProgramiv(program.get(), GL_INFO_LOG_LENGTH, &length);
		std::vector<GLchar> msg(length + 1);
		glGetProgramInfoLog(program.get(), length, 0, &msg[0]);
		std::cerr << "linking program failed.\n";
		std::cerr << (&msg[0]) << '\n';
		throw std::runtime_error("unable to link program");
	}

	// introspect program uniforms
	GLint numUniforms, maxUniformLength;
	glGetProgramiv(program.get(), GL_ACTIVE_UNIFORMS, &numUniforms);
	glGetProgramiv(program.get(), GL_ACTIVE_UNIFORM_MAX_LENGTH,
							&maxUniformLength);
	std::vector<GLchar> uname(maxUniformLength + 1);
	for (GLint i = 0; i != numUniforms; ++i) {
		GLint size;
		GLenum type;
		glGetActiveUniform(program.get(), i,
				maxUniformLength, NULL, &size, &type,
				&uname[0]);
		if (size != 1)
			std::cerr << "uniform arrays are not supported\n";
		GLint loc = glGetUniformLocation(program.get(),
							&uname[0]);
		ShaderVariable *sv = new ShaderVariable(this,
				std::string(&uname[0]), cvt_type(type));
		sv->location_ = loc;
		uniforms_.push_back(sv);
	}
	// TODO? remove uniforms whose location is -1

	// introspect vertex attributes
	GLint numAttribs, maxAttribLength;
	glGetProgramiv(program.get(), GL_ACTIVE_ATTRIBUTES, &numAttribs);
	glGetProgramiv(program.get(), GL_ACTIVE_ATTRIBUTE_MAX_LENGTH,
							&maxAttribLength);
	std::vector<GLchar> aname(maxAttribLength + 1);
	for (GLint i = 0; i != numAttribs; ++i) {
		GLint size;
		GLenum type;
		glGetActiveAttrib(program.get(), i,
				maxAttribLength, NULL, &size, &type,
				&aname[0]);
		if (size != 1)
			std::cerr << "attribute arrays are not supported\n";
		GLint loc = glGetAttribLocation(program.get(),
							&aname[0]);
		ShaderVariable *sv = new ShaderVariable(this,
				std::string(&aname[0]), cvt_type(type));
		sv->location_ = loc;
		attributes_.push_back(sv);
	}

#ifdef PRINT_SHADER_INFO
	{
		// print out all uniforms and vertex attributes
		std::map<GLenum, const char *> typeMap;
		typeMap[ShaderVariable::Float] = "float";
		typeMap[ShaderVariable::Vec2] = "float_vec2";
		typeMap[ShaderVariable::Vec3] = "float_vec3";
		typeMap[ShaderVariable::Vec4] = "float_vec4";
		typeMap[ShaderVariable::Int] = "int";
		typeMap[ShaderVariable::IVec2] = "int_vec2";
		typeMap[ShaderVariable::IVec3] = "int_vec3";
		typeMap[ShaderVariable::IVec4] = "int_vec4";
		typeMap[ShaderVariable::UInt] = "uint";
		typeMap[ShaderVariable::UVec2] = "uint_vec2";
		typeMap[ShaderVariable::UVec3] = "uint_vec3";
		typeMap[ShaderVariable::UVec4] = "uint_vec4";
		typeMap[ShaderVariable::Bool] = "bool";
		typeMap[ShaderVariable::BVec2] = "bool_vec2";
		typeMap[ShaderVariable::BVec3] = "bool_vec3";
		typeMap[ShaderVariable::BVec4] = "bool_vec4";
		typeMap[ShaderVariable::Mat2x2] = "float_mat2";
		typeMap[ShaderVariable::Mat3x3] = "float_mat3";
		typeMap[ShaderVariable::Mat4x4] = "float_mat4";
		typeMap[ShaderVariable::Mat2x3] = "float_mat2x3";
		typeMap[ShaderVariable::Mat3x2] = "float_mat3x2";
		typeMap[ShaderVariable::Mat2x4] = "float_mat2x4";
		typeMap[ShaderVariable::Mat4x2] = "float_mat4x2";
		typeMap[ShaderVariable::Mat3x4] = "float_mat3x4";
		typeMap[ShaderVariable::Mat4x3] = "float_mat4x3";

		std::cerr << "  program uniforms:\n";
		for (ShaderVariable *sv: uniforms_) {
			std::cerr << "    " << sv->name() << ' '
				<< ' ' << typeMap[sv->type()]
				<< " loc: " << sv->location() << '\n';
		}
		std::cerr << "  vertex attributes:\n";
		for (ShaderVariable *sv: attributes_) {
			std::cerr << "    " << sv->name() << ' '
				<< ' ' << typeMap[sv->type()]
				<< " loc: " << sv->location() << '\n';
		}
	}
#endif

	programObj = program.release();
}

ShaderProgram::~ShaderProgram()
{
	if (programObj)
		glDeleteProgram(programObj);
	if (vs)
		glDeleteShader(vs);
	if (fs)
		glDeleteShader(fs);
	for (ShaderVariable *sv: uniforms_) {
		delete sv;
	}
	for (ShaderVariable *sv: attributes_) {
		delete sv;
	}
}

ShaderVariable*
ShaderProgram::uniform(const string& name, bool exceptions) const
{
	for (ShaderVariable *sv: uniforms_) {
		if (sv->name() == name)
			return sv;
	}
	if (exceptions)
		throw std::runtime_error("shader uniform not found");
	return NULL;
}

ShaderVariable*
ShaderProgram::attribute(const string& name, bool exceptions) const
{
	for (ShaderVariable *sv: attributes_) {
		if (sv->name() == name)
			return sv;
	}
	if (exceptions)
		throw std::runtime_error("shader attribute not found");
	return NULL;
}

void
ShaderProgram::setup() const throw ()
{
	if (programObj == 0)
		return;

	glUseProgram(programObj);
	current_programObj = programObj;
	for (ShaderVariable *sv: uniforms_) {
		if (sv->location() == -1 || !sv->has_value())
			continue;
		sv->draw_uniform();
	}
}

void
ShaderProgram::cleanup() const throw ()
{
	if (programObj == 0)
		return;
	glUseProgram(0);
	current_programObj = 0;
}

} // namespace llgr
