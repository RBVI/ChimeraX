// Analytic computation of solvent accessible surface area, ie. the surface area of a union of spheres.

#include <math.h>		// use M_PI, sqrt, cos, atan2, ...

#include <iostream>		// use std::cerr for debugging
#include <vector>		// use std::vector

#include "pythonarray.h"	// use parse_double_n3_array, ...
#include "rcarray.h"		// use DArray

// Circle on a unit sphere specified by center point on sphere and angular radius.
class Circle
{
public:
  Circle(double center[3], double cos_angle)
  {
    for (int a = 0 ; a < 3 ; ++a)
      this->center[a] = center[a];
    this->cos_angle = cos_angle;
    this->angle = acos(cos_angle);
  }
  Circle(const Circle &c)
  {
    for (int a = 0 ; a < 3 ; ++a)
      this->center[a] = c.center[a];
    this->cos_angle = c.cos_angle;
    this->angle = c.angle;
  }
  double *centerp() { return &center[0]; }
  const double *centerp() const { return &center[0]; }
  double center[3], cos_angle, angle;
};

typedef std::vector<Circle> Circles;

class Circle_Intersection
{
public:
  Circle_Intersection(const Circle &circle1, const Circle &circle2, double point[3]) : circle1(&circle1), circle2(&circle2)
  {
    for (int a = 0 ; a < 3 ; ++a)
      this->point[a] = point[a];
  }
  ~Circle_Intersection() { circle1 = circle2 = NULL; }
  const Circle *circle1, *circle2;
  double point[3];
};

typedef std::vector<Circle_Intersection> Circle_Intersections;
typedef std::vector<Circle_Intersection*> Path;
typedef std::vector<Path> Paths;

static bool surface_area_of_spheres(double *centers, int n, double *radii, double *areas);
static bool buried_sphere_area(int i, double *centers, int n, double *radii, double *area);
static bool sphere_intersection_circles(int i, double *centers, int n, double *radii,
					Circles &circles);
static bool sphere_intersection(double *c0, double r0, double *c1, double r1,
				Circles &circles);
static bool area_in_circles_on_unit_sphere(const Circles &circles, double *area);
static void remove_circles_in_circles(const Circles &circles, Circles &circles2);
static int circle_intersections(const Circles &circles,
				Circle_Intersections &cint, Circles &lc);
static bool circle_in_circles(int i, const Circles &circles);
static bool circle_intercepts(const Circle &c0, const Circle &c1,
			      double p0[3], double p1[3]);
static bool point_in_circles(double p[3], const Circles &circles,
			     int iexclude0, int iexclude1);
static bool boundary_paths(Circle_Intersections &cint, Paths &paths);
static bool boundary_path(int i, Circle_Intersections &cint, bool *used, Path &path);
static double lone_circles_area(const Circles &lone_circles);
static double bounded_area(const Paths &paths, int nreg);
static double circle_intercept_angle(const double *center1, const double *pintersect, const double *center2);
static double polar_angle(const double *zaxis, const double *v1, const double *v2);
static void estimate_surface_area_of_spheres(double *centers, int n, double *radii,
					     double *sphere_points, int np, double *point_weights,
					     double *areas);
static double estimate_buried_sphere_area(int i, double *centers, int n, double *radii,
					  double *points, int np, double *weights, double wsum,
					  double *pbuf, int *ibuf);

class Region_Count
{
public:
  Region_Count(int n) : n(n)
  {
    this->c = new int[n];
    for (int i = 0 ; i < n ; ++i)
      c[i] = i;
  }
  ~Region_Count()
  {
    delete c;
    c = NULL;
  }
  void join(int i, int j)
  {
    int mi = min_connected(i), mj = min_connected(j);
    if (mi < mj)
      c[mj] = mi;
    else if (mj < mi)
      c[mi] = mj;
  }
  int min_connected(int i)
  {
    while (c[i] != i)
      i = c[i];
    return i;
  }
  int number_of_regions()
  {
    int nr = 0;
    for (int i = 0 ; i < n ; ++i)
      if (c[i] == i)
	nr += 1;
    return nr;
  }
  int *region_sizes()
  {
    int *s = new int[n];
    for (int i = 0 ; i < n ; ++i)
      s[i] = 0;
    for (int i = 0 ; i < n ; ++i)
      s[min_connected(i)] += 1;
    return s;
  }
private:
  int n;
  int *c;
};

static bool surface_area_of_spheres(double *centers, int n, double *radii, double *areas)
{
  for (int i = 0 ; i < n ; ++i)
    {
      double ba;
      if (!buried_sphere_area(i, centers, n, radii, &ba))
	return false;
      double r = radii[i];
      areas[i] = 4*M_PI*r*r - ba;
    }
  return true;
}

static bool buried_sphere_area(int i, double *centers, int n, double *radii, double *area)
{
  double r = radii[i];

  // Compute sphere intersections
  Circles circles;
  if (sphere_intersection_circles(i, centers, n, radii, circles))
    *area = 4*M_PI*r*r;   // Sphere is completely contained in another sphere
  else if (area_in_circles_on_unit_sphere(circles, area)) // Compute analytical buried area on sphere.
    *area *= r*r;
  else
    return false;	// Analytic calculation failed.

  return true;
}

// Returns true if sphere i is entirely inside another sphere.
static bool sphere_intersection_circles(int i, double *centers, int n, double *radii,
					Circles &circles)
{
  double *c = centers + 3*i, r = radii[i];
  for (int j = 0 ; j < n ; ++j)
    if (j != i)
      if (sphere_intersection(c, r, centers+3*j, radii[j], circles))
	return true;
  return false;
}

// Returns true if sphere 0 is entirely inside sphere 1.
static bool sphere_intersection(double *c0, double r0, double *c1, double r1,
				Circles &circles)
{
  double dx = c1[0]-c0[0], dy = c1[1]-c0[1], dz = c1[2]-c0[2];
  double d2 = dx*dx + dy*dy + dz*dz;
  if (d2 > (r0+r1)*(r0+r1))
    return false;		// Spheres don't intersect.
  double d = sqrt(d2);
  if (r0+d < r1)
    return true;		// Sphere 1 contains sphere 0
  if (r1+d < r0 || d == 0 || r0 == 0)
    return false;
  double ca = (r0*r0 + d*d - r1*r1) / (2*r0*d);
  if (ca < -1 || ca > 1)
    return false;
  double c[3] = {dx/d, dy/d, dz/d};
  circles.push_back(Circle(c, ca));
  return false;
}

static bool area_in_circles_on_unit_sphere(const Circles &circles, double *area)
{
  // Check if sphere is outside all other spheres.
  if (circles.size() == 0)
    return 0;

  // Speed up circle intersection calculation.
  // Typically half of circles are in other circles for molecular surfaces.
  Circles circles2;
  remove_circles_in_circles(circles, circles2);

  Circle_Intersections cint;
  Circles lc;
  int nreg = circle_intersections(circles2, cint, lc);
  //  std::cerr << cint.size() << " intersections " << lc.size() << " lone circles " << nreg << " regions\n";

  // Check if circles cover the sphere
  if (cint.empty() && lc.empty())
    {
      *area = 4*M_PI;
      return true;
    }

  // Connect circle arcs to form boundary paths.
  Paths paths;
  if (!boundary_paths(cint, paths))
    return false;

  /*
  std::cerr << "boundary lengths ";
  for (int p = 0 ; p < lc.size() ; ++p)
    std::cerr << "1,";
  for (int p = 0 ; p < paths.size() ; ++p)
    std::cerr << paths[p].size() << ",";
  if (nreg < paths.size())
    std::cerr << "for " << nreg + lc.size() << " region";
  std::cerr << std::endl;
  */

  double la = lone_circles_area(lc);
  double ba = bounded_area(paths, nreg);
  *area = la + ba;

  return true;
}

static void remove_circles_in_circles(const Circles &circles, Circles &circles2)
{
  // Remove circles contained in other circles.
  int nc = circles.size();
  for (int i = 0 ; i < nc ; ++i)
    if (!circle_in_circles(i, circles))
      circles2.push_back(circles[i]);
}

static int circle_intersections(const Circles &circles,
				Circle_Intersections &cint, Circles &lc)
{
  // Compute intersection points of circles that are not contained in other circles.
  int nc = circles.size();
  Region_Count rc(nc);
  for (int i = 0 ; i < nc ; ++i)
    {
      const Circle &c0 = circles[i];
      for (int j = i+1 ; j < nc ; ++j)
	{
	  const Circle &c1 = circles[j];
	  double p0[3], p1[3];
	  if (circle_intercepts(c0, c1, p0, p1))
	    {
	      rc.join(i,j);
	      if (!point_in_circles(p0, circles, i, j))
		cint.push_back(Circle_Intersection(c0,c1,p0));
	      if (!point_in_circles(p1, circles, i, j))
		cint.push_back(Circle_Intersection(c1,c0,p1));
	    }
	}
    }

  // Lone circles
  int *sz = rc.region_sizes();
  for (int i = 0 ; i < nc ; ++i)
    if (sz[i] == 1)
      lc.push_back(circles[i]);
  delete [] sz;

  // Number of multicircle regions
  int nreg = rc.number_of_regions() - lc.size();

  return nreg;
}

static bool circle_in_circles(int i, const Circles &circles)
{
  const double *p = circles[i].centerp(), a = circles[i].angle;
  int nc = circles.size();
  for (int j = 0 ; j < nc ; ++j)
    {
      const Circle &c = circles[j];
      const double *pj = c.centerp(), aj = c.angle;
      if (aj >= a && (p[0]*pj[0]+p[1]*pj[1]+p[2]*pj[2]) >= cos(aj-a) && j != i)
	return true;
    }
  return false;
}

static bool circle_intercepts(const Circle &c0, const Circle &c1,
			      double p0[3], double p1[3])
{
  double x0 = c0.center[0], y0 = c0.center[1], z0 = c0.center[2];
  double x1 = c1.center[0], y1 = c1.center[1], z1 = c1.center[2];
  double ca01 = x0*x1 + y0*y1 + z0*z1;
  double x01 = y0*z1-z0*y1, y01 = z0*x1-x0*z1, z01 = x0*y1-y0*x1;
  double s2 = x01*x01 + y01*y01 + z01*z01;
  if (s2 == 0)
    return false;
    
  double ca0 = c0.cos_angle;
  double ca1 = c1.cos_angle;
  double d2 = (s2 - ca0*ca0 - ca1*ca1 + 2*ca01*ca0*ca1);
  if (d2 < 0)
    return false;

  double a = (ca0 - ca01*ca1) / s2;
  double b = (ca1 - ca01*ca0) / s2;
  double d = sqrt(d2) / s2;

  double cx = a*x0 + b*x1, cy = a*y0 + b*y1, cz = a*z0 + b*z1;
  x01 *= d; y01 *= d; z01 *= d;
  p0[0] = cx - x01;  p0[1] = cy - y01;  p0[2] = cz - z01;
  p1[0] = cx + x01;  p1[1] = cy + y01;  p1[2] = cz + z01;
  return true;
}

static bool point_in_circles(double p[3], const Circles &circles,
			     int iexclude0, int iexclude1)
{
  int nc = circles.size();
  for (int i = 0 ; i < nc ; ++i)
    {
      const Circle &c = circles[i];
      const double *pi = c.center;
      if ((p[0]*pi[0]+p[1]*pi[1]+p[2]*pi[2]) >= c.cos_angle &&
	  i != iexclude0 && i != iexclude1)
	return true;
    }
  return false;
}

static bool boundary_paths(Circle_Intersections &cint, Paths &paths)
{
  int n = cint.size();
  bool *used = new bool[n];
  for (int i = 0 ; i < n ; ++i)
    used[i] = false;
  for (int i = 0 ; i < n ; ++i)
    if (!used[i])
      {
	paths.push_back(Path());
	if (!boundary_path(i, cint, used, paths.back()))
	  return false;
      }
  return true;
}

static bool boundary_path(int i, Circle_Intersections &cint, bool *used, Path &path)
{
  path.push_back(&cint[i]);
  int bp = i, n = cint.size();
  while (true)
    {
      int j_min = -1;
      double a_min = 3*M_PI;
      for (int j = 0 ; j < n ; ++j)
	{
	if (!used[j] && cint[j].circle1 == cint[bp].circle2)
	  {
	    double a = polar_angle(cint[bp].circle2->centerp(), cint[bp].point, cint[j].point);
	    if (a < a_min)
	      {
		a_min = a;
		j_min = j;
	      }
	  }
	}
      if (j_min == -1)
	return false;  // Could not follow boundary. Probably due to 3 or more circles intersecting at one point.
      if (j_min == i)
	break;
      path.push_back(&cint[j_min]);
      used[j_min] = true;
      bp = j_min;
    }
  used[i] = true;
  return true;
}

static double lone_circles_area(const Circles &lone_circles)
{
  double area = 0;
  int n = lone_circles.size();
  for (int i = 0 ; i < n ; ++i)
    area += 2*M_PI*(1-lone_circles[i].cos_angle);
  return area;
}

static double bounded_area(const Paths &paths, int nreg)
{
  double area = 0;
  int np = paths.size();
  for (int p = 0 ; p < np ; ++p)
    {
      const Path &path = paths[p];
      int n = path.size();
      double ba = 0;
      for (int i = 0 ; i < n ; ++i)
	{
	  Circle_Intersection *bp1 = path[i];
	  double *p = bp1->point;
	  const Circle *c1 = bp1->circle1, *c2 = bp1->circle2;
	  double ia = circle_intercept_angle(c1->centerp(), p, c2->centerp());
	  ba += ia - 2*M_PI;
	  Circle_Intersection *bp2 = path[(i+1)%n];
	  double a = polar_angle(c2->centerp(), p, bp2->point);  // Circular arc angle
	  //	  std::cerr << "seg " << i << " kink " << (ia - 2*M_PI)*180/M_PI << " arc " << a*180/M_PI << std::endl;
	  ba += a * bp1->circle2->cos_angle;  // circular segment bend angle
	}
      //      std::cerr << "path length " << n << " area " << 2*M_PI-ba << std::endl;
      area += 2*M_PI - ba;
    }
  if (np > nreg)
    area -= 4*M_PI*(np-nreg);
  return area;
}

static double circle_intercept_angle(const double *center1, const double *pintersect, const double *center2)
{
  // Angle made by tangent vectors t1 = c1 x p and t2 = c2 x p is same as
  // polar angle of c1 and c2 about p.
  double a = polar_angle(pintersect, center1, center2);
  return a;
}

// Angle from plane defined by z and v1 rotated to plane defined by z and v2
// range 0 to 2*pi.
static double polar_angle(const double *zaxis, const double *v1, const double *v2)
{
  double z0 = zaxis[0], z1 = zaxis[1], z2 = zaxis[2];
  double v10 = v1[0], v11 = v1[1], v12 = v1[2];
  double y0 = z1*v12-z2*v11, y1 = z2*v10-z0*v12, y2 = z0*v11-z1*v10;	// y = zaxis ^ v1
  double x0 = y1*z2-y2*z1, x1 = y2*z0-y0*z2, x2 = y0*z1-y1*z0;		// x = y ^ zaxis

  double xn = sqrt(x0*x0 + x1*x1 + x2*x2);
  double yn = sqrt(y0*y0 + y1*y1 + y2*y2);
  double v20 = v2[0], v21 = v2[1], v22 = v2[2];

  double x = v20*x0 + v21*x1 + v22*x2;
  double y = v20*y0 + v21*y1 + v22*y2;
  double a = atan2(xn*y,yn*x);
  if (a < 0)
    a += 2*M_PI;
  return a;
}

static void estimate_surface_area_of_spheres(double *centers, int n, double *radii,
					     double *sphere_points, int np, double *point_weights,
					     double *areas)
{
  double wsum = 0;
  for (int k = 0 ; k < np ; ++k)
    wsum += point_weights[k];

  double *pbuf = new double [3*np];
  int *ibuf = new int[np];

  for (int i = 0 ; i < n ; ++i)
    {
      double ba = estimate_buried_sphere_area(i, centers, n, radii,
					      sphere_points, np, point_weights, wsum,
					      pbuf, ibuf);
      double r = radii[i];
      areas[i] = 4*M_PI*r*r - ba;
    }

  delete [] ibuf;
  delete [] pbuf;
}

static double estimate_buried_sphere_area(int i, double *centers, int n, double *radii,
					  double *points, int np, double *weights, double wsum,
					  double *pbuf, int *ibuf)
{
  double *c = centers + 3*i, r = radii[i];
  for (int j = 0 ; j < np ; ++j)
    {
      for (int a = 0 ; a < 3 ; ++a)
	pbuf[3*j+a] = r*points[3*j+a] + c[a];
      ibuf[j] = 0;
    }

  double ci0 = c[0], ci1 = c[1], ci2 = c[2];
  for (int j = 0 ; j < n ; ++j)
    if (j != i)
      {
	double rj = radii[j];
	double c0 = centers[3*j], c1 = centers[3*j+1], c2 = centers[3*j+2];
	double d0 = c0-ci0, d1 = c1-ci1, d2 = c2-ci2;
	double dij2 = d0*d0 + d1*d1 + d2*d2;
	if (dij2 > (r+rj)*(r+rj))
	  continue;	// Spheres don't intersect.
	double r2 = rj*rj;
	for (int k = 0 ; k < np ; ++k)
	  if (!ibuf[k])
	    {
	      double *p = pbuf+3*k;
	      double dx = p[0]-c0, dy = p[1]-c1, dz = p[2]-c2;
	      double d2 = dx*dx + dy*dy + dz*dz;
	      if (d2 <= r2)
		ibuf[k] = 1;
	    }
      }

  double isum = 0;
  for (int k = 0 ; k < np ; ++k)
    if (ibuf[k])
      isum += weights[k];

  double a = 4*M_PI*r*r*isum/wsum;

  return a;
}

extern "C" PyObject *surface_area_of_spheres(PyObject *s, PyObject *args, PyObject *keywds)
{
  DArray centers, radii, areas;
  const char *kwlist[] = {"centers", "radii", "areas", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&|O&"), (char **)kwlist,
				   parse_double_n3_array, &centers,
				   parse_double_n_array, &radii,
				   parse_writable_double_n_array, &areas))
    return NULL;

  DArray ca = centers.contiguous_array();
  DArray ra = radii.contiguous_array();
  bool alloc_areas = (areas.dimension() == 0);
  if (alloc_areas)
    parse_writable_double_n_array(python_double_array(ca.size(0)), &areas);
  if (!areas.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "surface_area_of_spheres: area array must be contiguous");
      return NULL;
    }
  if (ra.size(0) != ca.size(0) || areas.size(0) != ca.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "surface_area_of_spheres: centers, radii and area arrays must be the same length.");
      return NULL;
    }

  bool success = surface_area_of_spheres(ca.values(), ca.size(0), ra.values(), areas.values());
  PyObject *py_areas = (success ? array_python_source(areas) : Py_None);
  if (!alloc_areas || !success)
    Py_INCREF(py_areas);    
  return py_areas;
}

extern "C" PyObject *estimate_surface_area_of_spheres(PyObject *s, PyObject *args, PyObject *keywds)
{
  DArray centers, radii, points, weights, areas;
  const char *kwlist[] = {"centers", "radii", "sphere_points", "point_weights", "areas", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&|O&"), (char **)kwlist,
				   parse_double_n3_array, &centers,
				   parse_double_n_array, &radii,
				   parse_double_n3_array, &points,
				   parse_double_n_array, &weights,
				   parse_writable_double_n_array, &areas))
    return NULL;

  DArray ca = centers.contiguous_array();
  DArray ra = radii.contiguous_array();
  DArray pa = points.contiguous_array();
  DArray wa = weights.contiguous_array();
  bool alloc_areas = (areas.dimension() == 0);
  if (alloc_areas)
    parse_writable_double_n_array(python_double_array(ca.size(0)), &areas);
  if (!areas.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "estimate_surface_area_of_spheres: area array must be contiguous");
      return NULL;
    }
  if (ra.size(0) != ca.size(0) || areas.size(0) != ca.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "estimate_surface_area_of_spheres: centers, radii and area arrays must be the same length.");
      return NULL;
    }
  if (pa.size(0) != wa.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "estimate_surface_area_of_spheres: sphere points and weights arrays must be the same length.");
      return NULL;
    }
  estimate_surface_area_of_spheres(ca.values(), ca.size(0), ra.values(),
				   pa.values(), pa.size(0), wa.values(),
				   areas.values());
  PyObject *py_areas = array_python_source(areas);
  if (!alloc_areas)
    Py_INCREF(py_areas);    
  return py_areas;
}
