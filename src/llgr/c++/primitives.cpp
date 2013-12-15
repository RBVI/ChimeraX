#include "llgr.h"
#include "llgr_int.h"
#include "spiral.h"
#include <string>
#include <map>
#include <math.h>

#ifndef M_PI
#define	 M_PI	3.14159265358979323846
#endif

namespace llgr {

namespace internal {

struct PrimitiveInfo {
	Id data_id;
	unsigned icount;	// index count
	Id index_id;
	DataType index_type;
	PrimitiveInfo(Id d, int ic, Id i, DataType(t)):
			data_id(d), icount(ic), index_id(i), index_type(t) {}
	PrimitiveInfo() {}
};

typedef std::map<int, PrimitiveInfo> ProtoGeom;

ProtoGeom proto_spheres;
ProtoGeom proto_cylinders;
ProtoGeom proto_cones;
ProtoGeom proto_fans;

void
build_sphere(const int N)
{
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
		GLfloat np[3];	// normal and position
	};
	npinfo *data = new npinfo [N];
	for (auto i = 0; i < N; ++i) {
		data[i].np[0] = static_cast<GLfloat>(pts[i].x);
		data[i].np[1] = static_cast<GLfloat>(pts[i].y);
		data[i].np[2] = static_cast<GLfloat>(pts[i].z);
	}
	Id data_id = --internal_buffer_id;
	create_buffer(data_id, ARRAY, N * sizeof (npinfo), data);
	delete [] data;

	Id index_id = --internal_buffer_id;
	create_buffer(index_id, ELEMENT_ARRAY, num_triangles * sizeof (spiral::tri_info), tris);

	proto_spheres[N] = PrimitiveInfo(data_id, 3 * num_triangles, index_id, UShort);
	delete [] pts;
	delete [] phis;
}

void
build_cylinder(const int N)
{
	if (N * 2 > 65535)
		throw std::runtime_error("too many vertices needed");
	struct npinfo {
		GLfloat n[3];	// normal
		GLfloat pos[3];	// position
	};
	npinfo *data = new npinfo [N * 2];
	unsigned num_indices = N * 2 + 2;
	GLushort *indices = new GLushort [num_indices];
	for (unsigned short i = 0; i < N; ++i) {
		float theta = 2.0f * float(M_PI) * i / N;
		float x = cosf(theta);
		float z = sinf(theta);
		data[i].n[0] = x;
		data[i].n[1] = 0;
		data[i].n[2] = z;
		data[i].pos[0] = x;
		data[i].pos[1] = -1;
		data[i].pos[2] = z;
		data[i + N] = data[i];
		data[i + N].pos[1] = 1;
		indices[i * 2] = i;
		indices[i * 2 + 1] = i + N;
	}
	indices[N * 2] = 0;
	indices[N * 2 + 1] = N;

	Id data_id = --internal_buffer_id;
	create_buffer(data_id, ARRAY, N * 2 * sizeof (npinfo), data);
	delete [] data;

	Id index_id = --internal_buffer_id;
	create_buffer(index_id, ELEMENT_ARRAY, num_indices * sizeof (GLushort), indices);
	proto_cylinders[N] = PrimitiveInfo(data_id, num_indices, index_id, UShort);
}

void
build_cone(const int N)
{
	if (N * 2 > 65535)
		throw std::runtime_error("too many vertices needed");
	struct npinfo {
		GLfloat n[3];	// normal
		GLfloat pos[3];	// position
	};
	npinfo *data = new npinfo [N * 3];
	unsigned num_indices = N * 2 + 2;
	GLushort *indices = new GLushort [num_indices];
	// TODO: the right normal, need to scale normals in vertex program?
	float y_normal = .5 / sqrtf(5);
	for (unsigned short i = 0; i < N; ++i) {
		float theta = 2.0f * float(M_PI) * i / N;
		float x = cosf(theta);
		float z = sinf(theta);
		data[i].n[0] = x;
		data[i].n[1] = y_normal;
		data[i].n[2] = z;
		data[i].pos[0] = x;
		data[i].pos[1] = -1;
		data[i].pos[2] = z;
		data[i + N] = data[i];
		data[i + N].pos[0] = 0;
		data[i + N].pos[1] = 1;
		data[i + N].pos[2] = 0;
		indices[i * 2] = i;
		indices[i * 2 + 1] = i + N;
	}
	indices[N * 2] = 0;
	indices[N * 2 + 1] = N;

	Id data_id = --internal_buffer_id;
	create_buffer(data_id, ARRAY, N * 2 * sizeof (npinfo), data);
	delete [] data;

	Id index_id = --internal_buffer_id;
	create_buffer(index_id, ELEMENT_ARRAY, num_indices * sizeof (GLushort), indices);
	proto_cones[N] = PrimitiveInfo(data_id, num_indices, index_id, UShort);
}

void
build_fan(const int N)
{
	if (N + 2 > 65535)
		throw std::runtime_error("too many vertices needed");
	auto data = new GLfloat [N + 2][3];
	data[0][0] = data[0][1] = data[0][2] = 0;
	for (auto i = 0; i < N; ++i) {
		float theta = 2.0f * float(M_PI) * i / N;
		float x = cosf(theta);
		float z = sinf(theta);
		data[N - i][0] = x;
		data[N - i][1] = 0;
		data[N - i][2] = z;
	}
	data[N + 1][0] = data[1][0];
	data[N + 1][1] = data[1][1];
	data[N + 1][2] = data[1][2];

	Id data_id = --internal_buffer_id;
	create_buffer(data_id, ARRAY, (N + 2) * 3 * sizeof (GLfloat), data);
	delete [] data;

	Id index_id = 0;	// no indices
	const unsigned num_indices = N + 2;

	proto_fans[N] = PrimitiveInfo(data_id, num_indices, index_id, UShort);
}

} // namespace internal;

using namespace internal;

inline void
clear_geom(ProtoGeom &geom)
{
	if (!all_buffers.empty()) {
		// not inside clear_buffers
		for (auto& i: geom) {
			const PrimitiveInfo &info = i.second;
			delete_buffer(info.data_id);
			if (info.index_id)
				delete_buffer(info.index_id);
		}
	}
	geom.clear();
}

void
clear_primitives()
{
	clear_geom(proto_spheres);
	clear_geom(proto_cylinders);
	clear_geom(proto_cones);
	clear_geom(proto_fans);
}

void
add_sphere(Id obj_id, float radius,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	int N = 300;	// TODO: make dependent on radius in pixels
	auto i = proto_spheres.find(N);
	if (i == proto_spheres.end())
		build_sphere(N);
	const PrimitiveInfo &pi = proto_spheres[N];
	AttributeInfos mai(ais);
	mai.push_back(AttributeInfo("normal", pi.data_id, 0, 12, 3, Float));
	mai.push_back(AttributeInfo("position", pi.data_id, 0, 12, 3, Float));
	Id scale_id = --internal_buffer_id;
	float scale[3] = { radius, radius, radius };
	create_singleton(scale_id, sizeof scale, scale);
	mai.push_back(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float));
	create_object(obj_id, program_id, matrix_id, mai, Triangles, 0,
					pi.icount, pi.index_id, pi.index_type);
}

void
add_cylinder(Id obj_id, float radius, float length,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	int N = 50;	// TODO: make dependent on radius in pixels
	auto i = proto_cylinders.find(N);
	if (i == proto_cylinders.end())
		build_cylinder(N);
	const PrimitiveInfo &pi = proto_cylinders[N];
	AttributeInfos mai(ais);
	mai.push_back(AttributeInfo("normal", pi.data_id, 0, 24, 3, Float));
	mai.push_back(AttributeInfo("position", pi.data_id, 12, 24, 3, Float));
	Id scale_id = --internal_buffer_id;
	float scale[3] = { radius, length / 2, radius };
	create_singleton(scale_id, sizeof scale, scale);
	mai.push_back(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float));
	create_object(obj_id, program_id, matrix_id, mai, Triangle_strip, 0,
					pi.icount, pi.index_id, pi.index_type);
}

void
add_cone(Id obj_id, float radius, float length,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	int N = 50;	// TODO: make dependent on radius in pixels
	auto i = proto_cones.find(N);
	if (i == proto_cones.end())
		build_cone(N);
	const PrimitiveInfo &pi = proto_cones[N];
	AttributeInfos mai(ais);
	mai.push_back(AttributeInfo("normal", pi.data_id, 0, 24, 3, Float));
	mai.push_back(AttributeInfo("position", pi.data_id, 12, 24, 3, Float));
	Id scale_id = --internal_buffer_id;
	float scale[3] = { radius, length / 2, radius };
	create_singleton(scale_id, sizeof scale, scale);
	mai.push_back(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float));
	create_object(obj_id, program_id, matrix_id, mai, Triangle_strip, 0,
					pi.icount, pi.index_id, pi.index_type);
}

void
add_disk(Id obj_id, float inner_radius, float outer_radius,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	// TODO: don't ignore inner radius
	int N = 50;	// TODO: make dependent on radius in pixels
	auto i = proto_fans.find(N);
	if (i == proto_fans.end())
		build_fan(N);
	const PrimitiveInfo &pi = proto_fans[N];
	AttributeInfos mai(ais);
	Id normal_id = --internal_buffer_id;
	float normal[3] = { 0, 1, 0 };
	create_singleton(normal_id, sizeof normal, normal);
	mai.push_back(AttributeInfo("normal", normal_id, 0, 0, 3, Float));
	mai.push_back(AttributeInfo("position", pi.data_id, 0, 0, 3, Float));
	Id scale_id = --internal_buffer_id;
	float scale[3] = { outer_radius, 1, outer_radius };
	create_singleton(scale_id, sizeof scale, scale);
	mai.push_back(AttributeInfo("instanceScale", scale_id, 0, 0, 3, Float));
	create_object(obj_id, program_id, matrix_id, mai, Triangle_fan, 0,
					pi.icount, pi.index_id, pi.index_type);
}

} // namespace
