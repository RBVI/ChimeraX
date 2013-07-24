#ifndef LLGR_INT_H
# define LLGR_INT_H

# define GLEW_NO_GLU
# include <GL/glew.h>
# include "llgr.h"
# include "ShaderProgram.h"
# include <stdint.h>
# include <string.h>
# include <assert.h>
# include <map>
# include <iostream>
# include <stdexcept>

# define USE_VAO

namespace llgr {

extern bool hasGLError(const char *message);

extern bool initialized;
extern void init();

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

extern const std::string& attribute_alias(const std::string& name);

extern void attr_location_info(ShaderVariable::Type type, unsigned *num_locations, unsigned *num_elements);
extern void setup_array_attribute(const BufferInfo &bi, const AttributeInfo &ai, int loc, unsigned num_locations);

class SingletonInfo {
	// Summarize information about singleton attributes that an object has.
	// Objects with the same program_id and the same singleton buffers can
	// be instanced if the singleton values are combined into a buffer.
public:
	SingletonInfo(DataType t, bool norm, unsigned char *buf, int loc, unsigned num_loc, unsigned num_elem):
		type(t), normalized(norm), data(buf),
		base_location(loc), num_locations(num_loc),
		num_elements(num_elem) {}
	DataType type;
	bool normalized;
	unsigned char *data;
	int base_location;
	unsigned num_locations;
	unsigned num_elements;

};
typedef std::vector<SingletonInfo> SingletonCache;

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
#ifdef USE_VAO
	mutable bool		cache_valid;
	mutable SingletonCache	singleton_cache;
	GLuint	vao;
#endif
	ObjectInfo(Id s, Id m, const AttributeInfos &a, PrimitiveType pt, unsigned f, unsigned c):
			program_id(s), matrix_id(m),
			hide(false), transparent(false), selected(false),
			ais(a), ptype(pt), first(f), count(c),
			index_buffer_id(0), index_buffer_type(Byte)
#ifdef USE_VAO
			, cache_valid(false), vao(0)
#endif
			{
			}
	ObjectInfo(Id s, Id m, const AttributeInfos &a, PrimitiveType pt, unsigned f, unsigned c, Id ib, DataType t):
			program_id(s), matrix_id(m),
			hide(false), transparent(false),
			ais(a), ptype(pt), first(f), count(c),
			index_buffer_id(ib), index_buffer_type(t)
#ifdef USE_VAO
			, cache_valid(false), vao(0)
#endif
			{
			}
	ObjectInfo() {}
#ifdef USE_VAO
	bool valid_cache() const { return cache_valid; }
	void invalidate_cache() { cache_valid = false; }
#endif
};

typedef std::map<Id, ObjectInfo*> AllObjects;
extern AllObjects all_objects;

struct AI_Name
{
	bool operator ()(const AttributeInfo &ai)
	{
		return ai.name == name;
	}
	AI_Name(const char *n): name(n) {}
	AI_Name(const std::string &n): name(n) {}
private:
	std::string name;
};

extern bool dirty;
extern void optimize();

} // namespace

#endif
