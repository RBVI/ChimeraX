#include "llgr_int.h"
#include "spiral.h"
#include <string>
#include <map>
#include <math.h>
#include <stdexcept>

#ifndef M_PI
#define	 M_PI	3.14159265358979323846
#endif

using std::string;

namespace {

typedef std::map<string, string> NameMap;
NameMap name_map;

void
init_name_map()
{
	name_map["position"] = "position";
	name_map["normal"] = "normal";
}

}

namespace llgr {

void
set_primitive_attribute_name(const string& name, const string& value)
{
	name_map[name] = value;
}

void build_sphere(float radius);
void build_cylinder(float radius);

struct PrimitiveInfo {
	Id data_id;
	unsigned icount;	// index count
	Id index_id;
	DataType index_type;
	PrimitiveInfo(Id d, int ic, Id i, DataType(t)):
			data_id(d), icount(ic), index_id(i), index_type(t) {}
	PrimitiveInfo() {}
};

typedef std::map<float, PrimitiveInfo> ProtoGeom;

Id uniform_scale_id;
ProtoGeom proto_spheres;
ProtoGeom proto_cylinders;

inline void
clear_geom(ProtoGeom &geom)
{
	if (!all_buffers.empty()) {
		// not inside clear_buffers
		for (ProtoGeom::iterator i = geom.begin(), e = geom.end();
								i != e; ++i) {
			const PrimitiveInfo &info = i->second;
			delete_buffer(info.data_id);
			delete_buffer(info.index_id);
		}
	}
	geom.clear();
}

void
clear_primitives()
{
	if (!all_buffers.empty()) {
		delete_buffer(uniform_scale_id);
	}
	uniform_scale_id = 0;
	clear_geom(proto_spheres);
	clear_geom(proto_cylinders);
}

void
add_sphere(Id obj_id, float radius,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	if (name_map.empty())
		init_name_map();
	ProtoGeom::iterator i = proto_spheres.find(radius);
	if (i == proto_spheres.end())
		build_sphere(radius);
	const PrimitiveInfo &pi = proto_spheres[radius];
	AttributeInfos mai(ais);
	mai.push_back(AttributeInfo(name_map["normal"],
				pi.data_id, 0, 24, 3, Float));
	mai.push_back(AttributeInfo(name_map["position"],
				pi.data_id, 12, 24, 3, Float));
	if (!uniform_scale_id) {
		uniform_scale_id = --internal_buffer_id;
		static const float uniform_scale[3] = { 1, 1, 1 };
		create_singleton(uniform_scale_id, sizeof uniform_scale, uniform_scale);
	}
	mai.push_back(AttributeInfo("instanceScale", uniform_scale_id, 0, 0, 3, Float));
	create_object(obj_id, program_id, matrix_id, mai, Triangles, 0,
					pi.icount, pi.index_id, pi.index_type);
}

void
add_cylinder(Id obj_id, float radius, float length,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	if (name_map.empty())
		init_name_map();
	ProtoGeom::iterator i = proto_cylinders.find(radius);
	if (i == proto_cylinders.end())
		build_cylinder(radius);
	const PrimitiveInfo &pi = proto_cylinders[radius];
	AttributeInfos mai(ais);
	mai.push_back(AttributeInfo(name_map["normal"],
				pi.data_id, 0, 24, 3, Float));
	mai.push_back(AttributeInfo(name_map["position"],
				pi.data_id, 12, 24, 3, Float));
	Id scale_id = --internal_buffer_id;
	float scale[3] = { 1, length / 2, 1 };
	create_singleton(scale_id, sizeof scale, scale);
	mai.push_back(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float));
	create_object(obj_id, program_id, matrix_id, mai, Triangle_strip, 0,
					pi.icount, pi.index_id, pi.index_type);
}

void
build_sphere(float radius)
{
	static const unsigned N = 60;	// TODO: LOD

	spiral::pt_info *pts = new spiral::pt_info [N];
	float *phis = new float [N];
	spiral::points(N, pts, phis, NULL);
	spiral::tri_info *tris = spiral::triangles(N, pts, phis);
	if (tris == NULL) {
		delete [] pts;
		delete [] phis;
		throw std::runtime_error("unable to build sphere");
	}
	unsigned int num_triangles = 2 * N - 4;

	struct npinfo {
		GLfloat n[3];	// normal
		GLfloat pos[3];	// position
	};
	npinfo *array = new npinfo [N];
	for (unsigned int i = 0; i < N; ++i) {
		array[i].n[0] = static_cast<GLfloat>(pts[i].x);
		array[i].n[1] = static_cast<GLfloat>(pts[i].y);
		array[i].n[2] = static_cast<GLfloat>(pts[i].z);
		array[i].pos[0] = static_cast<GLfloat>(radius * pts[i].x);
		array[i].pos[1] = static_cast<GLfloat>(radius * pts[i].y);
		array[i].pos[2] = static_cast<GLfloat>(radius * pts[i].z);
	}
	Id np = --internal_buffer_id;
	create_buffer(np, ARRAY, N * sizeof (npinfo), array);
	delete [] array;

	Id index_id = --internal_buffer_id;
	create_buffer(index_id, ELEMENT_ARRAY, num_triangles * sizeof (spiral::tri_info), tris);

	proto_spheres[radius] = PrimitiveInfo(np, 3 * num_triangles, index_id, UShort);
	delete [] pts;
	delete [] phis;
}

void
build_cylinder(float radius)
{
	static const unsigned N = 12;	// TODO: LOD
	struct npinfo {
		GLfloat n[3];	// normal
		GLfloat pos[3];	// position
	};
	npinfo *array = new npinfo [N * 2];
	unsigned num_indices = N * 2 + 2;
	GLushort *indices = new GLushort [num_indices];
	for (unsigned short i = 0; i < N; ++i) {
		float theta = 2.0f * float(M_PI) * i / N;
		float x = cosf(theta);
		float z = sinf(theta);
		array[i].n[0] = x;
		array[i].n[1] = 0;
		array[i].n[2] = z;
		array[i].pos[0] = radius * x;
		array[i].pos[1] = -1;
		array[i].pos[2] = radius * z;
		array[i + N] = array[i];
		array[i + N].pos[1] = 1;
		indices[i * 2] = i;
		indices[i * 2 + 1] = i + N;
	}
	indices[N * 2] = 0;
	indices[N * 2 + 1] = N;

	Id np = --internal_buffer_id;
	create_buffer(np, ARRAY, N * 2 * sizeof (npinfo), array);
	delete [] array;

	Id index_id = --internal_buffer_id;
	create_buffer(index_id, ELEMENT_ARRAY, num_indices * sizeof (GLushort), indices);
	proto_cylinders[radius] = PrimitiveInfo(np, N * 2 + 2, index_id, UShort);
}

} // namespace
