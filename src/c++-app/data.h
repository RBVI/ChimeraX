#ifndef DATA_H
#define DATA_H

struct Sphere {
	float	radius;
	float	center[3];
	float	color[4];
	bool operator<(const Sphere &s) const {
			return radius < s.radius;
		}
};
typedef struct Sphere Sphere;

struct Cylinder {
	float	radius;
	float	length;
	float	orient[3][4];
	float	color[4];
	bool operator<(const Cylinder &c) const {
			return radius < c.radius;
		}
};
typedef struct Cylinder Cylinder;

extern int width, height;
extern float projection_matrix[16], modelview_matrix[16];
extern float normal_matrix[9];

extern Sphere spheres[];
extern unsigned sphere_count;
extern Cylinder cylinders[];
extern unsigned cylinder_count;

#endif
