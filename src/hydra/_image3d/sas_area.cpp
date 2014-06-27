// Analytic computation of solvent accessible surface area, ie. the surface area of a union of spheres.

#include <math.h>		// use M_PI, sqrt, cos, atan2, ...
#include <stdlib.h>		// use drand48()

#include <iostream>		// use std::cerr for debugging
#include <vector>		// use std::vector

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
    this->sin_angle = sin(angle);
  }
  Circle(const Circle &c) { *this = c; }
  void operator=(const Circle &c)
  {
    for (int a = 0 ; a < 3 ; ++a)
      this->center[a] = c.center[a];
    this->cos_angle = c.cos_angle;
    this->sin_angle = c.sin_angle;
    this->angle = c.angle;
  }
  double *centerp() { return &center[0]; }
  const double *centerp() const { return &center[0]; }
  double center[3], cos_angle, sin_angle, angle;
};

typedef std::vector<Circle> Circles;

// A point of intersection between two circles on a sphere is used to
// trace the boundary of the buried area on the sphere.
class Circle_Intersection
{
public:
  Circle_Intersection(const Circle &circle1, const Circle &circle2, double point[3]) :
    circle1(&circle1), circle2(&circle2)
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

// List of indices into sphere list.
typedef std::vector<int> Index_List;

// To find spheres that intersect without testing every sphere for intersection
// with every other sphere, we compute a bounding box for the spheres and divide
// it in half successively until the boxes contain few spheres.  We keep track of
// the spheres entirely in the box and the additional ones that touch the box which
// may intersect those inside the box.  Then we do all-by-all sphere intersection
// tests for each box.
class Region_Spheres
{
public:
  Region_Spheres() {}
  Region_Spheres(double *centers, int n, double *radii);
  void split_region(double *centers, double *radii,
		    Region_Spheres &rs1, Region_Spheres &rs2) const;
  int longest_axis() const;
  void compute_region_bounds(double *centers, double *radii);
  void find_nearby_spheres(const Index_List &near, double *centers, double *radii);

  double xmin[3], xmax[3];
  Index_List in_region, near_region;
};

static int surface_area_of_spheres(double *centers, int n, double *radii, double *areas);
static void find_sphere_regions(double *centers, int n, double *radii, unsigned int max_size,
				std::vector<Region_Spheres> &rspheres);
static void subdivide_region(const Region_Spheres &rs, double *centers, double *radii,
			     unsigned int max_size, std::vector<Region_Spheres> &rspheres);
static bool buried_sphere_area(int i, const Index_List &iclose,
			       double *centers, double *radii, double *area);
static bool sphere_intersection_circles(int i, const Index_List &iclose,
					double *centers, double *radii,	Circles &circles);
static bool sphere_intersection(double *c0, double r0, double *c1, double r1,
				Circles &circles);
static bool area_in_circles_on_unit_sphere(Circles &circles, double *area);
static bool remove_circles_in_circles(Circles &circles);
static int circle_intersections(const Circles &circles,
				Circle_Intersections &cint, Circles &lc);
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
static double estimate_buried_sphere_area(int i, const Index_List &iclose, double *centers, double *radii,
					  double *points, int np, double *weights, double wsum,
					  double *pbuf, int *ibuf);

// Count connected regions composed of overlapping discs on a sphere.
// Need to know the number of regions because area calculation for regions with more than
// one bounding curve is different than with one bounding curve.
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

// Returns count of how many spheres calculation succeeded for.
// If calculation fails for a sphere the area for that sphere is set to -1.
static int surface_area_of_spheres(double *centers, int n, double *radii, double *areas)
{
  // Find spheres that intersect other spheres quickly by partitioning spheres into boxes.
  int max_spheres_in_region = 100;
  std::vector<Region_Spheres> rspheres;
  find_sphere_regions(centers, n, radii, max_spheres_in_region, rspheres);
  int c = 0;
  int nr = rspheres.size();

  // Parallelize loop over boxes containing sphere using OpenMP.
#pragma omp parallel shared(nr,rspheres,centers,radii,areas,c)
  {
    // Have each thread handle 100 boxes.
#pragma omp for schedule(dynamic,100)
  for (int r = 0 ; r < nr ; ++r)	// Loop over boxes of spheres.
    {
      Region_Spheres &rs = rspheres[r];
      Index_List &ir = rs.in_region;
      int nir = ir.size();
      for (int j = 0 ; j < nir ; ++j)	// For each sphere in box compute area.
	{
	  int i = ir[j];
	  double ba;
	  if (buried_sphere_area(i, rs.near_region, centers, radii, &ba))
	    {
	      double r = radii[i];
	      areas[i] = 4*M_PI*r*r - ba;
	      c += 1;
	    }
	  else
	    areas[i] = -1;	// Calculation failed.
	}
    }
  }
  return c;
}

// Subdivide bounding boxes to group nearby spheres.
static void find_sphere_regions(double *centers, int n, double *radii, unsigned int max_size,
				std::vector<Region_Spheres> &rspheres)
{
  Region_Spheres rs(centers, n, radii);
  subdivide_region(rs, centers, radii, max_size, rspheres);
}

static void subdivide_region(const Region_Spheres &rs, double *centers, double *radii,
			     unsigned int max_size, std::vector<Region_Spheres> &rspheres)
{
  if (rs.in_region.size() <= max_size)
    rspheres.push_back(rs);
  else
    {
      Region_Spheres rs1, rs2;
      rs.split_region(centers, radii, rs1, rs2);
      subdivide_region(rs1, centers, radii, max_size, rspheres);
      subdivide_region(rs2, centers, radii, max_size, rspheres);
    }
}

Region_Spheres::Region_Spheres(double *centers, int n, double *radii)
{
  for (int i = 0 ; i < n ; ++i)
    {
      in_region.push_back(i);
      near_region.push_back(i);
    }
  compute_region_bounds(centers, radii);
}

void Region_Spheres::split_region(double *centers, double *radii,
				  Region_Spheres &rs1, Region_Spheres &rs2) const
{
  // Divide region in half along axis a_max.
  int a = longest_axis();
  double xmid = 0.5*(xmax[a] + xmin[a]);
  int nir = in_region.size();
  for (int si = 0 ; si < nir ; ++si)
    {
      int i = in_region[si];
      if (centers[3*i+a] <= xmid)
	rs1.in_region.push_back(i);
      else
	rs2.in_region.push_back(i);
    }
  rs1.compute_region_bounds(centers, radii);
  rs2.compute_region_bounds(centers, radii);
  rs1.find_nearby_spheres(near_region, centers, radii);
  rs2.find_nearby_spheres(near_region, centers, radii);
}

int Region_Spheres::longest_axis() const
{
  // Find axis with maximum size.
  int a_max;
  double s_max = 0;
  for (int a = 0 ; a < 3 ; ++a)
    if (xmax[a]-xmin[a] > s_max)
      {
	s_max = xmax[a]-xmin[a];
	a_max = a;
      }
  return a_max;
}

void Region_Spheres::compute_region_bounds(double *centers, double *radii)
{
  // Compute bounds of spheres in region including radii.
  int nir = in_region.size();
  for (int si = 0 ; si < nir ; ++si)
    {
      int i = in_region[si];
      double *c = centers + 3*i, r = radii[i];
      for (int a = 0 ; a < 3 ; ++a)
	{
	  double x1 = c[a] - r, x2 = c[a] + r;
	  if (si == 0 || x2 > xmax[a])	xmax[a] = x2;
	  if (si == 0 || x1 < xmin[a])	xmin[a] = x1;
	}
    }
}

void Region_Spheres::find_nearby_spheres(const Index_List &near, double *centers, double *radii)
{
  int n = near.size();
  for (int ni = 0 ; ni < n ; ++ni)
    {
      int i = near[ni];
      double *c = centers+3*i, r = radii[i];
      bool far = false;
      for (int a = 0 ; a < 3 && !far ; ++a)
	far = (c[a]+r < xmin[a] || c[a]-r > xmax[a]);
      if (!far)
	near_region.push_back(i);
    }
}

static bool buried_sphere_area(int i, const Index_List &iclose,
			       double *centers, double *radii, double *area)
{
  double r = radii[i];

  // Compute sphere intersections
  Circles circles;
  if (sphere_intersection_circles(i, iclose, centers, radii, circles))
    *area = 4*M_PI*r*r;   // Sphere is completely contained in another sphere
  else if (area_in_circles_on_unit_sphere(circles, area)) // Compute analytical buried area on sphere.
    *area *= r*r;
  else
    return false;	// Analytic calculation failed.

  return true;
}

// Returns true if sphere i is entirely inside another sphere.
static bool sphere_intersection_circles(int i, const Index_List &iclose,
					double *centers, double *radii,	Circles &circles)
{
  double *c = centers + 3*i, r = radii[i];
  int n = iclose.size();
  for (int j = 0 ; j < n ; ++j)
    {
      int ic = iclose[j];
      if (ic != i)
	if (sphere_intersection(c, r, centers+3*ic, radii[ic], circles))
	  return true;
    }
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
  if (d2 == 0 && r0 == r1 && c0 > c1)
    return true;	// Identical spheres, only first copy is outside.
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

// Modifies circles list.
static bool area_in_circles_on_unit_sphere(Circles &circles, double *area)
{
  // Check if sphere is outside all other spheres.
  if (circles.size() == 0)
    {
      *area = 0;
      return true;
    }

  // Need to detect circles in circles for lone circle area calculation.
  if (remove_circles_in_circles(circles))
    {
      *area = 4*M_PI;	// Two discs overlap to cover sphere.
      return true;
    }

  // Compute intersections of all pairs of circles,
  // and also lone circles that don't intersect any other circles.
  Circle_Intersections cint;
  Circles lc;
  int nreg = circle_intersections(circles, cint, lc);

  // Check if circles cover the sphere
  if (cint.empty() && lc.empty())
    {
      *area = 4*M_PI;
      return true;
    }

  // Connect circle arcs to form boundary paths.
  Paths paths;
  if (!boundary_paths(cint, paths))
    return false;	// Boundary path failed to form loops.

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


// Remove discs entirely within other discs.  The circles list is modified.
// If a disc covers the complement of another disc so the whole sphere is covered
// by the two discs then return true, otherwise return false.
static bool remove_circles_in_circles(Circles &circles)
{
  for (unsigned int i = 0 ; i < circles.size() ; ++i)
    {
      const Circle &ci = circles[i];
      const double *pi = ci.centerp(), ai = ci.angle, cai = ci.cos_angle, sai = ci.sin_angle;
      for (unsigned int j = i+1 ; j < circles.size() ; j++)
	{
	  const Circle &cj = circles[j];
	  const double *pj = cj.centerp(), aj = cj.angle, caj = cj.cos_angle, saj = cj.sin_angle;
	  double pipj = pi[0]*pj[0]+pi[1]*pj[1]+pi[2]*pj[2];
	  if (pipj >= caj*cai+saj*sai)		// One disc contains the other.
	    {
	      if (aj >= ai)			// i inside j, remove i.
		{ circles[i] = circles.back(); circles.pop_back(); i -= 1; break; }
	      else				// j inside i, remove j.
		{ circles[j] = circles.back(); circles.pop_back(); j -= 1; }
	    }
	  else if (aj == ai && pj[0] == pi[0] && pj[1] == pi[1] && pj[2] == pi[2])
	    { circles[j] = circles.back(); circles.pop_back(); j -= 1; }  // Identical circles, remove j
	  else if (pipj < caj*cai-saj*sai && ai+aj >= M_PI)  // Each disc contains complement of the other.
	    return true;			// Entire sphere is covered by two discs.
	}
    }
  return false;
}

// Compute intersections of all pairs of circles,
// and also lone circles that don't intersect any other circles.
// Returns the number of connected regions formed by the intersecting circles.
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

// Check if a point is within any circle, other than the two circles
// whose intersection defined this point.
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

// Connect intersection points on boundary to form loops.
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

// Find a boundary path loop starting at circle intersection i.
// Record the intersections that were used in the "used" array.
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

// Return the sum of areas of non-overlapping circles.
static double lone_circles_area(const Circles &lone_circles)
{
  double area = 0;
  int n = lone_circles.size();
  for (int i = 0 ; i < n ; ++i)
    area += 2*M_PI*(1-lone_circles[i].cos_angle);
  if (area > 4*M_PI)
    area = 4*M_PI;	// circles cover entire sphere.
  return area;
}

// Return the area with a path defined by circular arcs.
// Surprisingly this equals 2*pi minus the angle of rotation of the
// boundary tangent vector as the bounding path is traversed.
// If more than one path bounds a region, then sum 2*pi minus
// tangent rotation over all loops and subtract 4*pi for each
// extra bounding path for a region.
static double bounded_area(const Paths &paths, int nreg)
{
  double area = 0;
  int np = paths.size();
  for (int p = 0 ; p < np ; ++p)
    {
      const Path &path = paths[p];
      int n = path.size();
      double ba = 0;	// Bend angle. Rotation of tangent vector on sphere surface.
      for (int i = 0 ; i < n ; ++i)
	{
	  Circle_Intersection *bp1 = path[i];
	  double *p = bp1->point;
	  const Circle *c1 = bp1->circle1, *c2 = bp1->circle2;
	  double ia = circle_intercept_angle(c1->centerp(), p, c2->centerp());
	  ba += ia - 2*M_PI;	// Tangent rotation at intersection of two arcs.
	  Circle_Intersection *bp2 = path[(i+1)%n];
	  double a = polar_angle(c2->centerp(), p, bp2->point);  // Circular arc angle
	  ba += a * bp1->circle2->cos_angle;  // Tangent rotation along circular arc
	}
      area += 2*M_PI - ba;
    }
  if (np > nreg)
    area -= 4*M_PI*(np-nreg);	// Correction for regions bounded by more than one loop.
  return area;
}

// Compute the tangent vector rotation at an intersection of two circles.
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

// Compute exposed surface area of set of spheres using a numerical approximation.
// Buried area for each sphere is computed by counting how many points on each sphere are
// contained in other spheres.  The points have area weights to handle non-uniformly
// distributed points.  One case uses icosahedral subdivision to produce the unit sphere
// points.
static void estimate_surface_area_of_spheres(double *centers, int n, double *radii,
					     double *sphere_points, int np, double *point_weights,
					     double *areas)
{
  // Find spheres that intersect other spheres quickly by partitioning spheres into boxes.
  int max_spheres_in_region = 100;
  std::vector<Region_Spheres> rspheres;
  find_sphere_regions(centers, n, radii, max_spheres_in_region, rspheres);
  int nr = rspheres.size();

  double wsum = 0;
  for (int k = 0 ; k < np ; ++k)
    wsum += point_weights[k];

  // Parallelize loop over boxes containing sphere using OpenMP.
#pragma omp parallel shared(nr,rspheres,centers,radii,sphere_points,np,point_weights,wsum,areas)
  {
  double *pbuf = new double [3*np];
  int *ibuf = new int[np];
  // Have each thread handle 100 boxes. 
#pragma omp for schedule(dynamic,100)
  for (int r = 0 ; r < nr ; ++r)	// Loop over boxes of spheres.
    {
      Region_Spheres &rs = rspheres[r];
      Index_List &ir = rs.in_region;
      int nir = ir.size();
      for (int j = 0 ; j < nir ; ++j)	// For each sphere in box compute area.
	{
	  int i = ir[j];
	  double ba = estimate_buried_sphere_area(i, rs.near_region, centers, radii,
						  sphere_points, np, point_weights, wsum,
						  pbuf, ibuf);
	  double r = radii[i];
	  areas[i] = 4*M_PI*r*r - ba;
	}
    }
  delete [] ibuf;
  delete [] pbuf;
  }
}

// Calculate area for sphere i buried by spheres iclose.
// Use points on unit sphere and point weights (areas) and see how many points are inside other spheres.
// pbuf and ibuf are buffers to hold points recentered and scaled to sphere i, and to mark which points
// were in other spheres.
static double estimate_buried_sphere_area(int i, const Index_List &iclose, double *centers, double *radii,
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
  int n = iclose.size();
  for (int k = 0 ; k < n ; ++k)
    {
      int j = iclose[k];
      if (j == i)
	continue;

      double rj = radii[j];
      double c0 = centers[3*j], c1 = centers[3*j+1], c2 = centers[3*j+2];
      double d0 = c0-ci0, d1 = c1-ci1, d2 = c2-ci2;
      double dij2 = d0*d0 + d1*d1 + d2*d2;
      if (dij2 > (r+rj)*(r+rj))
	continue;	// Spheres don't intersect.
      if (dij2 == 0 && rj == r)
	{
	  if (j > i)
	    continue;	// Identical spheres, only first is outside
	  else
	    return 4*M_PI*r*r;	// Rest are entirely buried.
	}
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

// Test program.
int main(int argc, char **argv)
{
  int n = 10;
  double *centers = new double[3*n], *radii = new double[n], *areas = new double[n];

  srand48(1345);	// Set random number seed.

  // Random sphere centers and radii.
  for (int i = 0 ; i < 3*n ; ++i)
    centers[i] = drand48();
  for (int i = 0 ; i < n ; ++i)
    radii[i] = 0.5*drand48();

  int ns = surface_area_of_spheres(centers, n, radii, areas);

  double a;
  for (int i = 0 ; i < n ; ++i)
    if (areas[i] >= 0)
      a += areas[i];

  std::cout << "Computed SAS area for " << n << " spheres, " << ns << " successful, total area = " << a << std::endl;

  int np = 10000;
  double *eareas = new double[n];
  double *sphere_points = new double[3*np], *point_weights = new double[np];

  // Uniformly distributed random points on unit sphere.
  for (int i = 0 ; i < 3*np ; )
    {
      double x = 2*drand48()-1, y = 2*drand48()-1, z = 2*drand48()-1;
      double d = sqrt(x*x + y*y + z*z);
      if (d <= 1)
	{ sphere_points[i++] = x/d; sphere_points[i++] = y/d; sphere_points[i++] = z/d; }
    }
  for (int i = 0 ; i < np ; ++i)
    point_weights[i] = 1;

  estimate_surface_area_of_spheres(centers, n, radii,
				   sphere_points, np, point_weights,
				   eareas);

  double ea;
  for (int i = 0 ; i < n ; ++i)
    ea += eareas[i];
  std::cout << " estimated with " << np << " points per sphere, total area = " << ea << std::endl;

  return 0;
}
