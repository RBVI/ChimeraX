#include "llgr_int.h"
#include "llgr_ui.h"
#include <math.h>
#include <stdexcept>

#define FUZZY_ZERO	1e-4
#define FUZZY_SQZERO	1e-8

namespace {

// subset of Vector math
class Vector {
public:
	double x, y, z;
	Vector() {}
	Vector(double a, double b, double c): x(a), y(b), z(c) {}
	double operator*(const Vector &r) const {
			return x * r.x + y * r.y + z * r.z;
		}
	double sqlength() const {
			return x * x + y * y + z * z;
		}
};

Vector
cross(const Vector &l, const Vector &r)
{
	Vector    result;

	result.x = (l.y * r.z) - (l.z * r.y);
	result.y = (l.z * r.x) - (l.x * r.z);
	result.z = (l.x * r.y) - (l.y * r.x);
	return result;
}

const double Identity[16] = {
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
};

struct VSphereInfo {
	double radius;
	double center[2];
	int x, y;
	llgr::VSphereCursor cursor;
};

typedef std::map<llgr::Id, VSphereInfo> AllVSpheres;

AllVSpheres all_vspheres;

inline double
normalize(double coord, double center, double radius)
{
	return (coord - center) / radius;
}

int
computeVSphere(double fx, double fy, double tx, double ty, double spin_axis[3], double *spin_angle)
{
	double d1 = fx * fx + fy * fy;
	double d2 = tx * tx + ty * ty;

	if (d1 > 1 && d2 < 1) {
		*spin_angle = 0;	// transition from z rotation to sphere rotation
		return 1;
	}
	if (d1 < 1 && d2 > 1) {
		*spin_angle = 0;	// transition from sphere rotation to z rotation
		return 2;
	}
	Vector from, to, axis;
	if (d1 < 1) {
		from = Vector(fx, fy, sqrt(1 - d1));
		to = Vector(tx, ty, sqrt(1 - d2));
	} else {
		d1 = sqrt(d1);
		d2 = sqrt(d2);
		from = Vector(fx / d1, fy / d1, 0);
		to = Vector(tx / d2, ty / d2, 0);
	}
	axis = cross(from, to);
	if (axis.sqlength() < FUZZY_SQZERO) {
		// if the two positions normalized to the same vector, punt.
		*spin_angle = 0;
		return 3;
	}
	double dot_product = from * to;	// from and to are "unit" length
	if (dot_product > 1) {
		// guarantee within acos bounds (more of a problem with floats)
		dot_product = 1;
	}
	*spin_angle = acos(dot_product);
	if (-FUZZY_ZERO < *spin_angle && *spin_angle < FUZZY_ZERO) {
		// may need to adjust threshold to avoid drift
		// on a per-input device basis
		*spin_angle = 0;
		return 4;
	}
	spin_axis[0] = axis.x;
	spin_axis[1] = axis.y;
	spin_axis[2] = axis.z;
	return 0;
}

} // namespace

namespace llgr {

void
vsphere_setup(Id vsphere, double radius, double center[2])
{
	VSphereInfo &vi = all_vspheres[vsphere];
	vi.radius = radius;
	vi.center[0] = center[0];
	vi.center[1] = center[1];
	// other fields are initialized in vsphere_press
}

VSphereCursor
vsphere_press(Id vsphere, int x, int y)
{
	AllVSpheres::iterator i = all_vspheres.find(vsphere);
	if (i == all_vspheres.end())
		throw std::logic_error("unknown vsphere");
	VSphereInfo &vi = i->second;

	vi.x = x;
	vi.y = y;
	double tx = normalize(x, vi.center[0], vi.radius);
	double ty = -normalize(y, vi.center[1], vi.radius);
	if (tx * tx + ty * ty >= 1)
		vi.cursor = ZRotation;
	else
		vi.cursor = Rotation;
	return vi.cursor;
}

VSphereCursor
vsphere_drag(Id vsphere, int x, int y, bool throttle, double spin_axis[3], double *spin_angle)
{
	AllVSpheres::iterator i = all_vspheres.find(vsphere);
	if (i == all_vspheres.end())
		throw std::logic_error("unknown vsphere");
	VSphereInfo &vi = i->second;

	double fx = normalize(vi.x, vi.center[0], vi.radius);
	double fy = -normalize(vi.y, vi.center[1], vi.radius);
	double tx = normalize(x, vi.center[0], vi.radius);
	double ty = -normalize(y, vi.center[1], vi.radius);

	switch (computeVSphere(fx, fy, tx, ty, spin_axis, spin_angle)) {
	  case 0:
		// normal case
		vi.x = x;
		vi.y = y;
		break;
	  case 1:
		// transition z-rotation to rotation
		vi.cursor = Rotation;
		vi.x = x;
		vi.y = y;
		break;
	  case 2:
		// transition rotation to z-rotation
		vi.cursor = ZRotation;
		vi.x = x;
		vi.y = y;
		break;
	  case 3:
		// from and to normalized to same point
		// don't update last x and y
		break;
	  case 4:
		// angle effectively zero
		// don't update last x and y
		break;
	}
	if (throttle)
		*spin_angle *= 0.1;
	return vi.cursor;
}

void
vsphere_release(Id vsphere)
{
	AllVSpheres::iterator i = all_vspheres.find(vsphere);
	if (i == all_vspheres.end())
		return;
	all_vspheres.erase(i);
}

} // namespace llgr
