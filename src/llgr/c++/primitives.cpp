#include "llgr_int.h"
#include "spiral.h"
#include <string>
#include <map>
#include <math.h>

#ifndef M_PI
#define	 M_PI	3.14159265358979323846
#endif

namespace llgr {

void build_sphere(int N);
void build_cylinder(int N);

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
	clear_geom(proto_spheres);
	clear_geom(proto_cylinders);
}

void
add_sphere(Id obj_id, float radius,
	Id program_id, Id matrix_id, const AttributeInfos& ais)
{
	int N = 300;	// TODO: make dependent on radius in pixels
	ProtoGeom::iterator i = proto_spheres.find(N);
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
	ProtoGeom::iterator i = proto_cylinders.find(N);
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
	for (unsigned int i = 0; i < N; ++i) {
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

} // namespace
