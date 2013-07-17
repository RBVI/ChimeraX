#include "llgr_int.h"
#include <stdexcept>
#include <string>
#include <sstream>

#define COMPAT_GL2

namespace llgr {

AllPrograms all_programs;
AllPrograms pick_programs;

const char pick_fragment_shader[] =
	"#version 150\n"
	"\n"
	"in vec4 f_pickId;\n"
	"\n"
	"out vec4 frag_color;\n"
	"\n"
	"void main (void)\n"
	"{\n"
	"  frag_color = f_pickId;\n"
	"}\n";

void
create_program(Id program_id, const char *vertex_shader, const char *fragment_shader, const char *pick_vertex_shader)
{
	if (!initialized)
		init();
	// TODO: if ever have a separate shader type, then overload with shaders
	if (program_id <= 0)
		throw std::runtime_error("need positive program id");
	std::string position = attribute_alias("position");
#if __APPLE__ && __MACH__
	glBindVertexArray(program_vao);
#endif
	ShaderProgram *sp = new ShaderProgram(vertex_shader, fragment_shader, position);
#if __APPLE__ && __MACH__
	glBindVertexArray(0);
#endif
	AllPrograms::iterator i = all_programs.find(program_id);
	if (i == all_programs.end()) {
		all_programs[program_id] = sp;
	} else {
		ShaderProgram *old_sp = i->second;
		i->second = sp;
		delete old_sp;
		i = pick_programs.find(program_id);
		if (i != pick_programs.end()) {
			old_sp = i->second;
			pick_programs.erase(i);
			delete old_sp;
		}
	}
	if (pick_vertex_shader == NULL)
		return;
#if __APPLE__ && __MACH__
	glBindVertexArray(program_vao);
#endif
	sp = new ShaderProgram(pick_vertex_shader, pick_fragment_shader, position);
#if __APPLE__ && __MACH__
	glBindVertexArray(0);
#endif
	pick_programs[program_id] = sp;
}

void
delete_program(Id program_id)
{
	AllPrograms::iterator i = all_programs.find(program_id);
	if (i == all_programs.end())
		return;
	ShaderProgram *sp = i->second;
	all_programs.erase(i);
	delete sp;
	i = pick_programs.find(program_id);
	if (i == pick_programs.end())
		return;
	sp = i->second;
	pick_programs.erase(i);
	delete sp;
}

void
clear_programs()
{
	AllPrograms save;

	all_programs.swap(save);
	for (AllPrograms::iterator i = save.begin(); i != save.end(); ++i) {
		ShaderProgram *sp = i->second;
		delete sp;
	}
	save.clear();
	pick_programs.swap(save);
	for (AllPrograms::iterator i = save.begin(); i != save.end(); ++i) {
		ShaderProgram *sp = i->second;
		delete sp;
	}
}

void
set_uniform(ShaderVariable *sv, ShaderType type, uint32_t data_length, Bytes data)
{
	int location = sv->location(); 
	if (location == -1) {
		// TODO: error saying using standard OpenGL functions for builtins
		return;
	}
	if (data_length != sv->byte_count()) {
		std::ostringstream os;
		os << "wrong number of bytes: expected " << sv->byte_count()
						<< " got " << data_length;
		throw std::logic_error(os.str());
	}
	switch (type) {
	  case FVec1: case FVec2: case FVec3: case FVec4:
	  case Mat2x2: case Mat3x3: case Mat4x4:
	  case Mat2x3: case Mat3x2:
	  case Mat2x4: case Mat4x2:
	  case Mat3x4: case Mat4x3:
		sv->setFloatv(static_cast<const GLfloat *>(data));
		break;
	  case IVec1: case IVec2: case IVec3: case IVec4:
	  case UVec1: case UVec2: case UVec3: case UVec4: sv->setIntv(static_cast<const GLint *>(data));
		break;
	}
}

void
set_uniform(Id program_id, const char *name, ShaderType type, uint32_t data_length, Bytes data)
{
	if (type >= Mat2x2) {
		set_uniform_matrix(program_id, name, false, type, data_length,
									data);
		return;
	}
	bool builtin = strncmp(name, "gl_", 3) == 0;
	if (!builtin != 0 && program_id == 0) {
		// broadcast to all current programs
		for (AllPrograms::iterator i = all_programs.begin(),
					e = all_programs.end(); i != e; ++i) {
			ShaderProgram *sp = i->second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform(sv, type, data_length, data);
		}
		for (AllPrograms::iterator i = pick_programs.begin(),
					e = pick_programs.end(); i != e; ++i) {
			ShaderProgram *sp = i->second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform(sv, type, data_length, data);
		}
		return;
	}
	if (!builtin) {
		ShaderProgram *sp = program_id ? all_programs[program_id] : NULL;
		ShaderVariable *sv = sp ? sp->uniform(name) : NULL;
		if (sv)
			set_uniform(sv, type, data_length, data);
		sp = program_id ? pick_programs[program_id] : NULL;
		sv = sp ? sp->uniform(name) : NULL;
		if (sv)
			set_uniform(sv, type, data_length, data);
		return;
	}
#ifdef COMPAT
	std::string n(name);
	assert(program_id == 0);
	if (n == "gl_Fog.color") {
		if (type == FVec4)
			glFogfv(GL_FOG_COLOR, static_cast<const GLfloat *>(data));
		else
			throw std::runtime_error("need FVec4 data");
		return;
	}
	// gl_Fog.scale is implicitly computed from gl_Fog.start and .end
	if (n == "gl_Fog.start") {
		if (type == FVec1)
			glFogfv(GL_FOG_START, static_cast<const GLfloat *>(data));
			throw std::runtime_error("need FVec1 data");
		return;
	}
	if (n == "gl_Fog.end") {
		if (type == FVec1)
			glFogfv(GL_FOG_END, static_cast<const GLfloat *>(data));
		else
			throw std::runtime_error("need FVec1 data");
		return;
	}
	if (n == "gl_LightModel.ambient") {
		if (type == FVec1)
			glLightModelfv(GL_LIGHT_MODEL_AMBIENT,
					static_cast<const GLfloat *>(data));
		else
			throw std::runtime_error("need FVec1 data");
		return;
	}
	static const size_t FM_LEN = sizeof "gl_FrontMaterial." - 1;
	if (n.substr(0, FM_LEN) == "gl_FrontMaterial.") {
		if (n.substr(FM_LEN + 2) == "specular") {
			if (type == FVec4)
				glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,
					static_cast<const GLfloat *>(data));
			else
				throw std::runtime_error("need FVec4 data");
		} else if (n.substr(FM_LEN + 2) == "shininess") {
			if (type == FVec1)
				glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS,
					static_cast<const GLfloat *>(data));
			else
				throw std::runtime_error("need FVec1 data");
		} else
			throw std::runtime_error("unsupported");
	}
	// gl_FrontMaterial.shininess 1 float loc:-1
	// gl_FrontLightProduct[0].specular 1 float_vec4 loc:-1
	// gl_FrontLightProduct[1].specular 1 float_vec4 loc:-1
	// gl_LightSource[0].diffuse 1 float_vec4 loc:-1
	// gl_LightSource[0].position 1 float_vec4 loc:-1
	// gl_LightSource[1].diffuse 1 float_vec4 loc:-1
	// gl_LightSource[1].position 1 float_vec4 loc:-1
	// gl_LightSource[2].diffuse 1 float_vec4 loc:-1
	// gl_LightSource[2].position 1 float_vec4 loc:-1
	static const size_t LS_LEN = sizeof "gl_LightSource[" - 1;
	if (n.substr(0, LS_LEN) == "gl_LightSource[") {
		GLenum light = GL_LIGHT0 + (n[LS_LEN] - '0');
		GLenum pname;
		if (n.substr(LS_LEN + 2) == "].diffuse")
			pname = GL_DIFFUSE;
		else if (n.substr(LS_LEN + 2) == "].specular")
			pname = GL_SPECULAR;
		else if (n.substr(LS_LEN + 2) == "].position")
			pname = GL_POSITION;
		else
			throw std::runtime_error("not supported");
		if (type == FVec4)
			glLightfv(light, pname,
					static_cast<const GLfloat *>(data));
		else
			throw std::runtime_error("need FVec4 data");
		return;
	}
#endif
	std::cerr << "warning: ignored uniform " << name << '\n';
}

void
set_uniform_matrix(ShaderVariable *sv, bool transpose, ShaderType type, uint32_t data_length, Bytes data)
{
	int location = sv->location();
	if (location == -1) {
		// TODO: error saying using standard OpenGL functions for builtins
		return;
	}
	if (data_length != sv->byte_count()) {
		std::ostringstream os;
		os << "wrong number of bytes: expected "
			<< sv->byte_count() << " got " << data_length;
		throw std::logic_error(os.str()); }
	switch (type) {
	  default:
		throw std::logic_error("need matrix type");
	  case Mat2x2: case Mat3x3: case Mat4x4:
#if 0
	  case Mat2x3: case Mat3x2:
	  case Mat2x4: case Mat4x2:
	  case Mat3x4: case Mat4x3:
#endif
		// TODO
		sv->setFloatMatrixv(transpose, static_cast<const GLfloat *>(data));
		break;
	}
}

void
set_uniform_matrix(Id program_id, const char *name, bool transpose, ShaderType type, uint32_t data_length, Bytes data)
{
	bool builtin = strncmp(name, "gl_", 3) == 0;
	if (builtin) {
		std::cerr << "warning: ignored uniform " << name << '\n';
		return;
	}
	if (program_id == 0) {
		// broadcast to all current programs
		for (AllPrograms::iterator i = all_programs.begin(),
					e = all_programs.end(); i != e; ++i) {
			ShaderProgram *sp = i->second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform_matrix(sv, transpose, type,
							data_length, data);
		}
		for (AllPrograms::iterator i = pick_programs.begin(),
					e = pick_programs.end(); i != e; ++i) {
			ShaderProgram *sp = i->second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform_matrix(sv, transpose, type,
							data_length, data);
		}
		return;
	}

	ShaderProgram *sp = program_id ? all_programs[program_id] : NULL;
	ShaderVariable *sv = sp ? sp->uniform(name) : NULL;
	set_uniform_matrix(sv, transpose, type, data_length, data);
	sp = program_id ? pick_programs[program_id] : NULL;
	sv = sp ? sp->uniform(name) : NULL;
	if (sv)
		set_uniform_matrix(sv, transpose, type, data_length, data);
	sp = program_id ? pick_programs[program_id] : NULL;
	sv = sp ? sp->uniform(name) : NULL;
	if (sv)
		set_uniform_matrix(sv, transpose, type, data_length, data);
}

} // namespace
