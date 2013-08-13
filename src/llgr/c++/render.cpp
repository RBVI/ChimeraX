#include "llgr_int.h"
#include "llgr_ui.h"
#include "limits.h"
#include <algorithm>

#undef PICK_DEBUG

// state-sorting scene graph of arrays and instances
//
// glVertexAttributePointer per attribute (glVertexAttrib if singleton)
// sort by program, then attribute
// automatic instancing: if attributes identical except for singletons,
//   then they can be combined into a private attribute array
// TODO: figure out how hard it would be to combine objects that used
// adjacent parts of buffers (e.g., objects that corresponded to the
// individual residue parts of a ribbon)

namespace llgr {

float clear_color[4];

void
set_clear_color(float r, float g, float b, float a)
{
	clear_color[0] = r;
	clear_color[1] = g;
	clear_color[2] = b;
	clear_color[3] = a;
	glClearColor(r, g, b, a);
}

GLenum
cvt_DataType(DataType dt)
{
	switch (dt) {
	  case Byte: return GL_BYTE;
	  case UByte: return GL_UNSIGNED_BYTE;
	  case Short: return GL_SHORT;
	  case UShort: return GL_UNSIGNED_SHORT;
	  case Int: return GL_INT;
	  case UInt: return GL_UNSIGNED_INT;
	  case Float: return GL_FLOAT;
	}
	return 0;
}

size_t
data_size(DataType type)
{
	switch (type) {
	  case Byte: return 1;
	  case UByte: return 1;
	  case Short: return 2;
	  case UShort: return 2;
	  case Int: return 4;
	  case UInt: return 4;
	  case Float: return 4;
	}
	return 0;
}

void
attr_location_info(ShaderVariable::Type type, unsigned *num_locations, unsigned *num_elements)
{
	// return the number of attribute locations used for the shader
	// variable (think matrices), and the number of elements in each
	// location.
	typedef ShaderVariable SV;
	switch (type) {
	  case SV::Float: case SV::Int: case SV::UInt: case SV::Bool:
		*num_locations = *num_elements = 1;
		return;
	  case SV::Vec2: case SV::IVec2: case SV::UVec2: case SV::BVec2:
		*num_locations = 1; *num_elements = 2;
		return;
	  case SV::Vec3: case SV::IVec3: case SV::UVec3: case SV::BVec3:
		*num_locations = 1; *num_elements = 3;
		return;
	  case SV::Vec4: case SV::IVec4: case SV::UVec4: case SV::BVec4:
		*num_locations = 1; *num_elements = 4;
		return;
	  case SV::Mat2x2:
		*num_locations = 2; *num_elements = 2;
		return;
	  case SV::Mat2x3:
		*num_locations = 2; *num_elements = 3;
		return;
	  case SV::Mat2x4:
		*num_locations = 2; *num_elements = 4;
		return;
	  case SV::Mat3x3:
		*num_locations = 3; *num_elements = 3;
		return;
	  case SV::Mat3x2:
		*num_locations = 3; *num_elements = 2;
		return;
	  case SV::Mat3x4:
		*num_locations = 3; *num_elements = 4;
		return;
	  case SV::Mat4x4:
		*num_locations = 4; *num_elements = 4;
		return;
	  case SV::Mat4x2:
		*num_locations = 4; *num_elements = 2;
		return;
	  case SV::Mat4x3:
		*num_locations = 4; *num_elements = 3;
		return;
#ifdef HAVE_TEXTURE
		// TODO
		Sampler1D, Sampler2D, Sampler3D, SamplerCube, 
		Sampler1DShadow, Sampler2DShadow,
#endif
	  default:
		*num_locations = *num_elements = 0;
		return;
	}
}

struct Attr_Name
{
	bool operator ()(const ShaderVariable *sv)
	{
		return sv->name() == name;
	}
	Attr_Name(const char *n): name(n) {}
	Attr_Name(const std::string &n): name(n) {}
private:
	std::string name;
};

void
setup_array_attribute(const BufferInfo &bi, const AttributeInfo &ai, int loc, unsigned num_locations)
{
	// TODO? do we need more than one VAPointer for arrays of matrices?
	GLenum type = cvt_DataType(ai.type);
	glBindBuffer(bi.target, bi.buffer);
	// TODO: if shader variable is int, use glVertexAttribIPointer
	glVertexAttribPointer(loc, ai.count, type, ai.normalized,
			ai.stride, reinterpret_cast<char *>(ai.offset));
	for (unsigned i = 0; i != num_locations; ++i)
		glEnableVertexAttribArray(loc + i);
}

void
setup_singleton_attribute(unsigned char *data, DataType type, bool normalized, int loc, unsigned num_locations, unsigned num_elements)
{

	if (type != Float) {
		static bool did_once;
		if (!did_once) {
			std::cerr << "WebGL only supports float singleton vertex attributes\n";
			did_once = true;
		}
	}

	size_t size = num_elements * data_size(type);
	for (unsigned i = 0; i != num_locations; ++i) {
		switch (num_elements) {
		  case 1:
			switch (type) {
			  case Byte: case UByte: break;
			  case Short: case UShort: glVertexAttrib1sv(loc, reinterpret_cast<GLshort *>(data)); break;
			  case Int: case UInt: break;
			  case Float: glVertexAttrib1fv(loc, reinterpret_cast<GLfloat *>(data)); break;
			}
			break;
		  case 2:
			switch (type) {
			  case Byte: case UByte: break;
			  case Short: case UShort: glVertexAttrib2sv(loc, reinterpret_cast<GLshort *>(data)); break;
			  case Int: case UInt: break;
			  case Float: glVertexAttrib2fv(loc, reinterpret_cast<GLfloat *>(data)); break;
			}
			break;
		  case 3:
			switch (type) {
			  case Byte: case UByte: break;
			  case Short: case UShort: glVertexAttrib3sv(loc, reinterpret_cast<GLshort *>(data)); break;
			  case Int: case UInt: break;
			  case Float: glVertexAttrib3fv(loc, reinterpret_cast<GLfloat *>(data)); break;
			}
			break;
		  case 4:
			switch (type) {
			  case Byte:
				if (normalized)
				     glVertexAttrib4Nbv(loc, reinterpret_cast<GLbyte *>(data));
				else
				     glVertexAttrib4bv(loc, reinterpret_cast<GLbyte *>(data));
				break;
			  case UByte:
				if (normalized)
					glVertexAttrib4Nubv(loc, reinterpret_cast<GLubyte *>(data));
				else
					glVertexAttrib4ubv(loc, reinterpret_cast<GLubyte *>(data));
				break;
			  case Short:
				if (normalized)
					glVertexAttrib4Nsv(loc, reinterpret_cast<GLshort *>(data));
				else
					glVertexAttrib4sv(loc, reinterpret_cast<GLshort *>(data));
				break;
			  case UShort:
				if (normalized)
					glVertexAttrib4Nusv(loc, reinterpret_cast<GLushort *>(data));
				else
					glVertexAttrib4usv(loc, reinterpret_cast<GLushort *>(data));
				break;
			  case Int:
				if (normalized)
					glVertexAttrib4Niv(loc, reinterpret_cast<GLint *>(data));
				else
					glVertexAttrib4iv(loc, reinterpret_cast<GLint *>(data));
				break;
			  case UInt:
				if (normalized)
					glVertexAttrib4Nuiv(loc, reinterpret_cast<GLuint *>(data));
				else
					glVertexAttrib4uiv(loc, reinterpret_cast<GLuint *>(data));
				break;
			  case Float: glVertexAttrib4fv(loc, reinterpret_cast<GLfloat *>(data)); break;
			}
			break;
		}
		data += size;
		loc += 1;
	}
}

#if 1
void
setup_attribute(ShaderProgram *sp, const AttributeInfo &ai)
{
	typedef ShaderProgram::Variables Vars;
	const Vars &attrs = sp->attributes();
	Vars::const_iterator i = std::find_if(attrs.begin(), attrs.end(),
							Attr_Name(ai.name));
	if (i == attrs.end())
		return;
	ShaderVariable *sv = *i;
	int loc = sv->location();

	AllBuffers::iterator bii = all_buffers.find(ai.data_id);
	if (bii == all_buffers.end())
		return;
	const BufferInfo &bi = bii->second;

	if (loc == -1) {
		throw std::runtime_error("builtin attributes are not supported");
		return;
	}

	unsigned num_locations;
	unsigned num_elements;
	attr_location_info(sv->type(), &num_locations, &num_elements);

	if (bi.buffer != 0) {
		setup_array_attribute(bi, ai, loc, num_locations);
	} else {
		setup_singleton_attribute(bi.data, ai.type, ai.normalized, loc, num_locations, num_elements);
		for (unsigned i = 0; i != num_locations; ++i)
			glDisableVertexAttribArray(loc + i);
	}
}
#endif

#ifdef PICK_DEBUG
int pick_x, pick_y;
#endif

void
render()
{
	if (!initialized)
		init();
#if __APPLE__ && __MACH__
	static GLuint default_vao;
	if (!default_vao) {
		// using glVertexAttribPointer fails unless a VAO is bound
		glGenVertexArrays(1, &default_vao);
		glBindVertexArray(default_vao);
	}
#endif
#ifdef PICK_DEBUG
	(void) pick(pick_x, pick_y);
	return;
#endif
	if (dirty)
		optimize();

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_DITHER);
	glDisable(GL_SCISSOR_TEST);

	if (all_objects.empty())
		return;

	ShaderProgram *sp = NULL;
	Id current_program_id = 0;
	Id current_matrix_id = INT_MAX;
	// instance transform singleton info
	DataType it_type = Float;
	unsigned char *it_data = NULL;
	int it_loc = -1;
	unsigned it_locations, it_elements;
	attr_location_info(ShaderVariable::Mat4x4, &it_locations, &it_elements);
	// TODO: only for opaque objects
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	for (AllObjects::iterator i = all_objects.begin(),
					e = all_objects.end(); i != e; ++i) {
		ObjectInfo *oi = i->second;
		if (oi->hide || !oi->program_id)
			continue;
		// setup program
		if (oi->program_id != current_program_id) {
			AllPrograms::iterator i
					= all_programs.find(oi->program_id);
			if (sp)
				sp->cleanup();
			if (i == all_programs.end()) {
				sp = NULL;
				continue;
			}
			sp = i->second;
			sp->setup();
			current_program_id = oi->program_id;
			current_matrix_id = INT_MAX;
			ShaderVariable *sv = sp->attribute("instanceTransform");
			it_loc = sv->location();
		}
		if (sp == NULL)
			continue;
#ifdef USE_VAO
		glBindVertexArray(oi->vao);
#else
		// setup index buffer
		const BufferInfo *ibi = NULL;
		if (oi->index_buffer_id) {
			AllBuffers::const_iterator bii
				= all_buffers.find(oi->index_buffer_id);
			if (bii == all_buffers.end())
				continue;
			ibi = &bii->second;
		}
#endif
		// setup instance matrix attribute
		if (oi->matrix_id != current_matrix_id) {
			Id data_id;
			if (oi->matrix_id == 0) {
				data_id = 0;
			} else {
				AllMatrices::iterator mii
					= all_matrices.find(oi->matrix_id);
				if (mii == all_matrices.end())
					continue;
				const MatrixInfo &mi = mii->second;
				data_id = mi.data_id;
			}
			AllBuffers::iterator bii = all_buffers.find(data_id);
			if (bii == all_buffers.end())
				continue;
			const BufferInfo &bi = bii->second;
			it_data = bi.data;
			setup_singleton_attribute(it_data, it_type, false, it_loc, it_locations, it_elements);
			current_matrix_id = oi->matrix_id;
		}
#ifdef USE_VAO
		for (SingletonCache::iterator sci = oi->singleton_cache.begin(); sci != oi->singleton_cache.end(); ++sci) {
			const SingletonInfo &si = *sci;
			setup_singleton_attribute(si.data, si.type, si.normalized, si.base_location, si.num_locations, si.num_elements);
		}
#else
		// setup other attributes
		for (AttributeInfos::iterator aii = oi->ais.begin();
						aii != oi->ais.end(); ++aii) {
			setup_attribute(sp, *aii);
		}
#endif
		// finally draw object
#ifndef USE_VAO
		if (!ibi) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#else
		if (!oi->index_buffer_id) {
#endif
			glDrawArrays(oi->ptype, oi->first, oi->count);
		} else {
#ifndef USE_VAO
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibi->buffer);
#endif
			glDrawElements(oi->ptype, oi->count,
				cvt_DataType(oi->index_buffer_type),
				reinterpret_cast<char *>(oi->first
					* data_size(oi->index_buffer_type)));
		}
#ifndef USE_VAO
		// cleanup
		// TODO: minimize cleanup between objects
		typedef ShaderProgram::Variables Vars;
		const Vars &attrs = sp->attributes();
		for (Vars::const_iterator svi = attrs.begin(); svi != attrs.end(); ++svi) {
			ShaderVariable *sv = *svi;
			int loc = sv->location();
			if (loc == -1)
				continue;
			glDisableVertexAttribArray(loc);
			// TODO: might be matrix, so disable more than one
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
	}
#ifdef USE_VAO
	glBindVertexArray(0);
#else
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#endif
	if (sp)
		sp->cleanup();
}

Id
pick(int x, int y)
{
	// Just like rendering except the color is monotonically increasing
	// and varies by object, not within an object.
	// Assume WebGL defaults of 8-bits each for red, green, and blue,
	// for a maximum of 16,777,215 (2^24 - 1) objects and that object ids
	// are also less than 16,777,215.
	// Note: could relax assumption and do multipass algorithm that
	// dicards color buffer for each pass and does object ids modulo
	// 2^24-1 + 1, but we're not anticipating needing that.
	if (dirty)
		optimize();

#ifndef PICK_DEBUG
	// render 5x5 pixels around hot spot
	glScissor(x - 2, y - 2, 5, 5);
	glEnable(GL_SCISSOR_TEST);
#else
	pick_x = x;
	pick_y = y;
#endif

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_DITHER);
	glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

	ShaderProgram *sp = NULL;
	Id current_program_id = 0;
	Id current_matrix_id = INT_MAX;
	AttributeInfo matrix_ai("instanceTransform", 0, 0, 0, 16, Float);
	Id pick_buffer_id = --internal_buffer_id;
	create_singleton(pick_buffer_id, 4, NULL);
	uint32_t *pick_id = NULL;
	{
		AllBuffers::const_iterator bii
				= all_buffers.find(pick_buffer_id);
		assert(bii != all_buffers.end());
		const BufferInfo &bi = bii->second;
		pick_id = reinterpret_cast<uint32_t *>(bi.data);
	}
	*pick_id = 0;
	AttributeInfo pick_ai("pickId", pick_buffer_id, 0, 0, 4, UByte, true);

	// TODO: only for opaque objects
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	for (AllObjects::iterator i = all_objects.begin(),
					e = all_objects.end(); i != e; ++i) {
		Id oid = i->first;
		ObjectInfo *oi = i->second;
		if (oi->hide || !oi->program_id)
			continue;
		// setup program
		if (oi->program_id != current_program_id) {
			AllPrograms::iterator i
					= pick_programs.find(oi->program_id);
			if (sp)
				sp->cleanup();
			if (i == pick_programs.end()) {
				sp = NULL;
				continue;
			}
			sp = i->second;
			sp->setup();
			current_program_id = oi->program_id;
			current_matrix_id = INT_MAX;
		}
		if (sp == NULL)
			continue;
		// setup index buffer
		const BufferInfo *ibi = NULL;
		if (oi->index_buffer_id) {
			AllBuffers::const_iterator bii
				= all_buffers.find(oi->index_buffer_id);
			if (bii == all_buffers.end())
				continue;
			ibi = &bii->second;
		}
		// setup instance matrix attribute
		if (oi->matrix_id != current_matrix_id) {
			if (oi->matrix_id == 0) {
				matrix_ai.data_id = 0;
			} else {
				AllMatrices::iterator mii
					= all_matrices.find(oi->matrix_id);
				if (mii == all_matrices.end())
					continue;
				const MatrixInfo &mi = mii->second;
				matrix_ai.data_id = mi.data_id;
			}
			setup_attribute(sp, matrix_ai);
			current_matrix_id = oi->matrix_id;
		}
		// setup pick id
#ifndef PICK_DEBUG
		*pick_id = oid;	// little-endian, alpha byte is msb
#else
static uint32_t colors[8] = { 0x90, 0xf0, 0x9000, 0xf000, 0x900000, 0xf00000, 0x909090, 0xf0f0f0 };
		*pick_id = colors[oid % 8];
#endif
		setup_attribute(sp, pick_ai);
		// setup other attributes
		for (AttributeInfos::iterator aii = oi->ais.begin();
						aii != oi->ais.end(); ++aii) {
			setup_attribute(sp, *aii);
		}
		// finally draw object
		if (!ibi) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glDrawArrays(oi->ptype, oi->first, oi->count);
		} else {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibi->buffer);
			glDrawElements(oi->ptype, oi->count,
				cvt_DataType(oi->index_buffer_type),
				reinterpret_cast<char *>(oi->first
					* data_size(oi->index_buffer_type)));
		}
		// cleanup
		// TODO: minimize cleanup between objects
		typedef ShaderProgram::Variables Vars;
		const Vars &attrs = sp->attributes();
		for (Vars::const_iterator svi = attrs.begin(); svi != attrs.end(); ++svi) {
			ShaderVariable *sv = *svi;
			int loc = sv->location();
			if (loc == -1)
				continue;
			glDisableVertexAttribArray(loc);
			// TODO: might be matrix, so disable more than one
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	if (sp)
		sp->cleanup();
	delete_buffer(pick_buffer_id);
	++internal_buffer_id;
	uint32_t data[5][5];
	glReadPixels(x - 2, y - 2, 5, 5, GL_RGBA, GL_UNSIGNED_BYTE, data);
#ifdef PICK_DEBUG
	for (int i = 0; i < 5; ++i)
		data[i][0] = data[i][4] = ~0;
	for (int i = 1; i < 4; ++i)
		data[0][i] = data[4][i] = ~0;
	glWindowPos2i(x - 2, y - 2);
	glDrawPixels(5, 5, GL_RGBA, GL_UNSIGNED_BYTE, data);
	std::cerr << "pick result: " << (data[2][2] & 0xffffff) << '\n';
#endif
	return data[2][2] & 0xffffff;
}

} // namespace
