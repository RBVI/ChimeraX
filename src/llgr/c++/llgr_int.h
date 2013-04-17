#ifndef LLGR_INT_H
# define LLGR_INT_H

# define GLEW_NO_GLU
# include <GL/glew.h>
# include "llgr.h"
# include "ShaderProgram.h"
# include <string.h>
# include <assert.h>
# include <map>
# include <iostream>

namespace llgr {

extern bool hasGLError(const char *message);

extern size_t data_size(DataType type);

typedef std::map<Id, ShaderProgram *> AllPrograms;
extern AllPrograms all_programs;
extern AllPrograms pick_programs;

struct BufferInfo
{
	GLuint	buffer;
	GLenum	target;
	unsigned size;		// only if singleton
	unsigned char *data;	// only if singleton data
	int	offset;		// only if singleton index
	BufferInfo(): buffer(0), target(0), data(NULL), offset(0) {}
	BufferInfo(GLuint b, GLenum ta):
		buffer(b), target(ta), size(0), data(NULL), offset(0) {}
	BufferInfo(GLenum ta, unsigned s, unsigned char *d):
		buffer(0), target(ta), size(s), data(d), offset(0) {}
};

typedef std::map<Id, BufferInfo> AllBuffers;
extern AllBuffers all_buffers;

extern Id internal_buffer_id;	// decrement before using

struct MatrixInfo {
	Id	data_id;
	bool	renormalize;
	MatrixInfo(Id d, bool r): data_id(d), renormalize(r) {}
	MatrixInfo() {}
};

typedef std::map<Id, MatrixInfo> AllMatrices;
extern AllMatrices all_matrices;

struct ObjectInfo {
	Id	program_id;
	Id	matrix_id;
	bool	hide;
	bool	transparent;
	bool	selected;
	AttributeInfos ais;
	PrimitiveType ptype;
	unsigned first, count;
	Id	index_buffer_id;
	DataType index_buffer_type;
	ObjectInfo(Id s, Id m, const AttributeInfos &a, PrimitiveType pt, unsigned f, unsigned c):
			program_id(s), matrix_id(m),
			hide(false), transparent(false), selected(false),
			ais(a), ptype(pt), first(f), count(c),
			index_buffer_id(0), index_buffer_type(Byte) {}
	ObjectInfo(Id s, Id m, const AttributeInfos &a, PrimitiveType pt, unsigned f, unsigned c, Id ib, DataType t):
			program_id(s), matrix_id(m),
			hide(false), transparent(false),
			ais(a), ptype(pt), first(f), count(c),
			index_buffer_id(ib), index_buffer_type(t) {}
	ObjectInfo() {}
};

typedef std::map<Id, ObjectInfo*> AllObjects;
extern AllObjects all_objects;

struct AI_Name
{
	bool operator ()(const AttributeInfo &ai)
	{
		return ai.name == name;
	}
	std::string name;
	AI_Name(const char *n): name(n) {}
	AI_Name(const std::string &n): name(n) {}
};

extern bool dirty;
extern void optimize();

} // namespace

#endif
