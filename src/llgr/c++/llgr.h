#ifndef LLGR_H
# define LLGR_H

// low-level graphics library

#include <vector>
#include <string>
#include <stdint.h>

# ifndef LLGR_DLL
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define LLGR_IMEX __attribute__((__visibility__("default")))
#  else
#   define LLGR_IMEX
#  endif
# elif defined(LLGR_EXPORT)
#  define LLGR_IMEX __declspec(dllexport)
# else
#  define LLGR_IMEX __declspec(dllimport)
# endif

# ifdef WrapPy
// stdint.h equivalents
typedef int int32_t;
typedef unsigned uint32_t;
# endif

namespace llgr {

#ifndef WrapPy
typedef const void * const Bytes;
#else
typedef const unsigned char * const Bytes;
#endif

typedef int Id;	// identifier provided to back-end
		// (negative values are internal to back-end)

enum DataType {
	// Double is not supported by OpenGL ES 2.0/3.0 nor WebGL 1.0
	Byte, UByte, Short, UShort, Int, UInt, Float
};

// programs

enum ShaderType {
	IVec1, IVec2, IVec3, IVec4,
	UVec1, UVec2, UVec3, UVec4,		// OpenGL ES 3 placeholder
	FVec1, FVec2, FVec3, FVec4,
	Mat2x2, Mat3x3, Mat4x4,
	Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3	// ditto
};

// value member is C++ type corresponding to ShaderType
template <typename T>
struct TypeToShaderType;

// Shapder programs

LLGR_IMEX extern void create_program(Id program_id, const char *vertex_shader, const char *fragment_shader, const char *pick_vertex_shader = NULL);
LLGR_IMEX extern void delete_program(Id program_id);
LLGR_IMEX extern void clear_programs();

// for set_uniform, program_id of zero means all programs with given uniform
LLGR_IMEX extern void set_uniform(Id program_id, const char *name, ShaderType type, uint32_t data_length, Bytes data);
template <typename T>
inline void set_uniform(Id program_id, const char *name, const T &data);
LLGR_IMEX extern void set_uniform_matrix(Id program_id, const char *name, bool transpose, ShaderType type, uint32_t data_length, Bytes data);

// (interleaved) buffer support

enum BufferTarget {
	ARRAY = 0x8892,		// same as GL_ARRAY_BUFFER
	ELEMENT_ARRAY = 0x8893	// same as GL_ELEMENT_ARRAY_BUFFER
};

// Create buffer of array data
LLGR_IMEX extern void create_buffer(Id data_id, BufferTarget target, uint32_t data_length, Bytes data);
LLGR_IMEX extern void delete_buffer(Id data_id);
LLGR_IMEX extern void clear_buffers();

// create singleton "buffer" data
LLGR_IMEX extern void create_singleton(Id data_id, uint32_t data_length, Bytes data);
#if 0
// TODO
LLGR_IMEX extern void create_singleton_index(Id data_id, Id reference_data_id, uint32_t size, uint32_t offset);

// update column of existing buffer data
LLGR_IMEX extern void update_buffer(Id data_id, uint32_t offset, uint32_t stride, uint32_t data_length, Bytes data);
#endif

#if 0
// TODO: textures and volume support
//
enum TextureFormat {
	RGB, RGBA, Luminance, LuminanceAlpha
};
enum TextureFilter {
	Nearest, Linear, NearestMimapNearest, NearestMipmapNearest,
	LinearMimapNearest, LinearMipmapLinear
};
LLGR_IMEX extern void create_2d_texture(Id tex_id, TextureFormat format, TextureFilter min_filter, TextureFilter max_filter, DataType type, uint32_t width, uint32_t height, Bytes data);
LLGR_IMEX extern void create_3d_texture(Id tex_id, TextureFormat format, TextureFilter min_filter, TextureFilter max_filter, DataType type, uint32_t width, uint32_t height, uint32_t depth, Bytes data);
LLGR_IMEX extern void create_2d_texture(Id tex_id, TextureFilter min_filter, TextureFilter max_filter, const char *relative_url);
LLGR_IMEX extern void delete_texture(Id data_id);
LLGR_IMEX extern void clear_textures();
#endif

// matrices

// instance matrix support
// TODO? just use a singleton buffer instead and default attribute values

// matrix_id of zero is reserved for identity matrix
// TODO: provide a second matrix with the corresponding normal matrix?
// renormalize should be true when the rotation part of the matrix
// has shear or scaling, or if it is a projection matrix. 
LLGR_IMEX extern void create_matrix(Id matrix_id, const float matrix[4][4], bool renormalize  = false);
LLGR_IMEX extern void delete_matrix(Id matrix_id);
LLGR_IMEX extern void clear_matrices();

// flat scene graph

struct AttributeInfo {
	std::string name;
	Id	data_id;
	uint32_t offset;	// byte offset into buffer
	uint32_t stride;	// byte stride to next element in buffer
	uint32_t count;		// number of data type
	DataType type;
	bool	normalized;	// only for integer types
	AttributeInfo(const std::string& n, Id d, uint32_t o, uint32_t s, uint32_t c, DataType t, bool norm = false): name(n), data_id(d), offset(o), stride(s), count(c), type(t), normalized(norm) {}
};
typedef std::vector<AttributeInfo> AttributeInfos;

enum PrimitiveType {
	Points = 0,		// same as GL_POINT
	Lines = 1,		// same as GL_LINES
	Line_loop = 2,		// same as GL_LINE_LOOP
	Line_strip = 3,		// same as GL_LINE_STRIP
	Triangles = 4,		// same as GL_TRIANGLES
	Triangle_strip = 5,	// same as GL_TRIANGLE_STRIP
	Triangle_fan = 6,	// same as GL_TRIANGLE_FAN
};

LLGR_IMEX extern const std::string& attribute_alias(const std::string& name);
LLGR_IMEX extern void set_attribute_alias(const std::string& name,
	const std::string& value);

LLGR_IMEX extern void create_object(Id obj_id, Id program_id, Id matrix_id,
	const AttributeInfos& ais, PrimitiveType pt,
	uint32_t first, uint32_t count,
	Id index_data_id = 0, DataType index_data_type = Byte);
// LLGR_IMEX extern void remove_object(Id obj_id, Id matrix_id);
LLGR_IMEX extern void delete_object(Id obj_id);
LLGR_IMEX extern void clear_objects();

#ifdef WrapPy
typedef std::vector<Id> Objects;
#endif

// indicate whether to draw object or not
#ifndef WrapPy
template <typename _iterable> void hide_objects(const _iterable& objects);
template <typename _iterable> void show_objects(const _iterable& objects);
#else
LLGR_IMEX extern void hide_objects(const Objects& objects);
LLGR_IMEX extern void show_objects(const Objects& objects);
#endif

// indicate whether an object is transparent or opaque (default opaque)
#ifndef WrapPy
template <typename _iterable> void transparent(const _iterable& objects);
template <typename _iterable> void opaque(const _iterable& objects);
#else
LLGR_IMEX extern void transparent(const Objects& objects);
LLGR_IMEX extern void opaque(const Objects& objects);
#endif

// indicate whether an object is selected or not
#ifndef WrapPy
template <typename _iterable> void selection_add(const _iterable& objects);
template <typename _iterable> void selection_remove(const _iterable& objects);
#else
LLGR_IMEX extern void selection_add(const Objects& objects);
LLGR_IMEX extern void selection_remove(const Objects& objects);
#endif
LLGR_IMEX extern void selection_clear();

// groups

LLGR_IMEX extern void create_group(Id group_id);
LLGR_IMEX extern void delete_group(Id group_id, bool and_objects=false);
LLGR_IMEX extern void clear_groups(bool and_objects=false);
#ifndef WrapPy
template <typename _iterable> void group_add(Id group_id, const _iterable& objects);
template <typename _iterable> void group_remove(Id group_id, const _iterable& objects);
#else
LLGR_IMEX extern void group_add(Id group_id, const Objects& objects);
LLGR_IMEX extern void group_remove(Id group_id, const Objects& objects);
#endif
LLGR_IMEX extern void hide_group(Id group_id);
LLGR_IMEX extern void show_group(Id group_id);
LLGR_IMEX extern void selection_add_group(Id group_id);
LLGR_IMEX extern void selection_remove_group(Id group_id);

// TODO: text primitives

// LOD primitives

LLGR_IMEX extern void add_sphere(Id obj_id, float radius,
	Id program_id, Id matrix_id, const AttributeInfos& ais);
LLGR_IMEX extern void add_cylinder(Id obj_id, float radius, float length,
	Id program_id, Id matrix_id, const AttributeInfos& ais);
LLGR_IMEX extern void clear_primitives();

// misc

LLGR_IMEX extern void clear_all();

LLGR_IMEX extern void set_clear_color(float red, float green, float blue,
								float alpha);

LLGR_IMEX extern void render();

} // namespace llgr

# include "llgr_impl.h"

#endif
