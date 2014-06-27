#include "llgr.h"
#include "llgr_int.h"
#include "ShaderProgram.h"
#include <string>
#include <sstream>

namespace llgr {

namespace internal {

AllPrograms all_programs;
AllPrograms pick_programs;

const char pick_fragment_shader[] =
	"#version 150\n"
	"\n"
	"in vec3 f_pickId;\n"
	"\n"
	"out vec3 frag_color;\n"
	"\n"
	"void main (void)\n"
	"{\n"
	"  frag_color = f_pickId;\n"
	"}\n";

} // namespace internal

using namespace internal;

void
create_program(Id program_id, const char *vertex_shader, const char *fragment_shader, const char *pick_vertex_shader)
{
	if (!initialized)
		init();
	// TODO: if ever have a separate shader type, then overload with shaders
	if (program_id <= 0)
		throw std::runtime_error("need positive program id");
	std::string position = attribute_alias("position");
	ShaderProgram *sp = new ShaderProgram(vertex_shader, fragment_shader, position);
	delete_program(program_id);
	all_programs[program_id] = sp;
	if (pick_vertex_shader == NULL)
		return;
	sp = new ShaderProgram(pick_vertex_shader, pick_fragment_shader, position);
	pick_programs[program_id] = sp;
}

void
delete_program(Id program_id)
{
	auto i = all_programs.find(program_id);
	if (i == all_programs.end())
		return;
	ShaderProgram *sp = i->second;
	all_programs.erase(i);
	delete sp;
	for (auto& i: all_objects) {
		ObjectInfo *oi = i.second;
		if (oi->program_id == program_id)
			oi->invalidate_cache();
	}
	// TODO: delete VAO?
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
	for (auto& i: save) {
		ShaderProgram *sp = i.second;
		delete sp;
	}
	save.clear();
	pick_programs.swap(save);
	for (auto& i: save) {
		ShaderProgram *sp = i.second;
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
		sv->set_floatv(static_cast<const GLfloat *>(data));
		break;
	  case IVec1: case IVec2: case IVec3: case IVec4:
	  case UVec1: case UVec2: case UVec3: case UVec4: sv->set_intv(static_cast<const GLint *>(data));
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
	if (builtin) {
		std::cerr << "warning: ignored uniform " << name << '\n';
		return;
	}
	if (program_id == 0) {
		// broadcast to all current programs
		for (auto& i: all_programs) {
			ShaderProgram *sp = i.second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform(sv, type, data_length, data);
		}
		for (auto& i: pick_programs) {
			ShaderProgram *sp = i.second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform(sv, type, data_length, data);
		}
		return;
	}
	ShaderProgram *sp = all_programs[program_id];
	ShaderVariable *sv = sp ? sp->uniform(name) : NULL;
	if (sv)
		set_uniform(sv, type, data_length, data);
	sp = pick_programs[program_id];
	sv = sp ? sp->uniform(name) : NULL;
	if (sv)
		set_uniform(sv, type, data_length, data);
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
		sv->set_float_matrixv(transpose, static_cast<const GLfloat *>(data));
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
		for (auto& i: all_programs) {
			ShaderProgram *sp = i.second;
			ShaderVariable *sv = sp->uniform(name);
			if (sv)
				set_uniform_matrix(sv, transpose, type,
							data_length, data);
		}
		for (auto& i: pick_programs) {
			ShaderProgram *sp = i.second;
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
