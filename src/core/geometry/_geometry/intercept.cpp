// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

static bool closest_triangle_intercept(const float *varray,
				       const int *tarray, int nt,
				       const float *xyz1, const float *xyz2,
				       float *fmin, int *tmin);
static bool triangle_intercept(const float *va, const float *vb,
			       const float *vc,
			       const float *xyz1, const float *xyz2,
			       float *fret);
static bool closest_sphere_intercept(const float *centers, int n, int cstride0, int cstride1,
				     const float *radii, int rstride,
				     const float *xyz1, const float *xyz2,
				     float *fmin, int *snum);
static bool closest_cylinder_intercept(const float *base1, int n, int b1stride0, int b1stride1,
				       const float *base2, int b2stride0, int b2stride1,
				       const float *radii, int rstride,
				       const float *xyz1, const float *xyz2,
				       float *fmin, int *cnum);

const char *closest_triangle_intercept_doc = 
  "closest_triangle_intercept(vertices, tarray, xyz1, xyz2) -> f, tnum\n"
  "\n"
  "Find first triangle intercept along line segment from xyz1 to xyz2.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "vertices : n by 3 float array\n"
  "  triangle vertex x,y,z coordinates.\n"
  "triangles : m by 3 int array\n"
  "  vertex indices specifying 3 vertices for each triangle.\n"
  "xyz1, xyz2 : float 3-tuples\n"
  "  x,y,z coordinates of line segment endpoints.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "f : float\n"
  "  fraction of distance from xyz1 to xyz2.  None if no intercept.\n"
  "tnum : int\n"
  "  triangle number, or None if no intercept.\n";

// ----------------------------------------------------------------------------
// Find closest triangle intercepting line segment between xyz1 and xyz2.
// The vertex array is xyz points (n by 3, NumPy single).
// The triangle array is triples of indices into the vertex array (m by 3,
// NumPy intc).
// Returns fraction of way along segment triangle index.
//
extern "C"
PyObject *closest_triangle_intercept(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray vertices;
  IArray triangles;
  float xyz1[3], xyz2[3];
  const char *kwlist[] = {"vertices", "triangles", "xyz1", "xyz2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &vertices,
				   parse_int_n3_array, &triangles,
				   parse_float_3_array, &xyz1,
				   parse_float_3_array, &xyz2))
    return NULL;

  float fmin;
  int tmin;
  PyObject *py_fmin, *py_tmin;
  if (closest_triangle_intercept(vertices.values(), triangles.values(), triangles.size(0),
				 xyz1, xyz2, &fmin, &tmin))
    {
      py_fmin = PyFloat_FromDouble(fmin);
      py_tmin = PyLong_FromLong(tmin);
    }
  else
    {
      py_fmin = Py_None; Py_INCREF(Py_None);
      py_tmin = Py_None; Py_INCREF(Py_None);
    }
  PyObject *t = python_tuple(py_fmin, py_tmin);

  return t;
}

// ----------------------------------------------------------------------------
//
static bool closest_triangle_intercept(const float *varray,
				       const int *tarray, int nt,
				       const float *xyz1, const float *xyz2,
				       float *fmin, int *tmin)
{
  float fc = -1;
  int tc = -1;
  for (int t = 0 ; t < nt ; ++t)
    {
      int ia = tarray[3*t], ib = tarray[3*t+1], ic = tarray[3*t+2];
      const float *va = varray + 3*ia, *vb = varray + 3*ib, *vc = varray + 3*ic;
      float f;
      if (triangle_intercept(va, vb, vc, xyz1, xyz2, &f))
	if (f < fc || fc < 0)
	  {
	    fc = f;
	    tc = t;
	  }
    }
  if (fc < 0)
    return false;

  *fmin = fc;
  *tmin = tc;
  return true;
}

// ----------------------------------------------------------------------------
//
static bool triangle_intercept(const float *va, const float *vb,
			       const float *vc,
			       const float *xyz1, const float *xyz2,
			       float *fret)
{
  float ba[3] = {vb[0]-va[0], vb[1]-va[1], vb[2]-va[2]};
  float ca[3] = {vc[0]-va[0], vc[1]-va[1], vc[2]-va[2]};
  float t[3] = {xyz2[0]-xyz1[0], xyz2[1]-xyz1[1], xyz2[2]-xyz1[2]};
  float p[3] = {xyz1[0]-va[0], xyz1[1]-va[1], xyz1[2]-va[2]};
  float n[3] = {ba[1]*ca[2]-ba[2]*ca[1],	// Normal to triangle.
		ba[2]*ca[0]-ba[0]*ca[2],
		ba[0]*ca[1]-ba[1]*ca[0]};
  
  float pn = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
  float tn = t[0]*n[0]+t[1]*n[1]+t[2]*n[2];
  if (tn == 0)
    return false;	// Segment is parallel to triangle plane.

  float f = -pn/tn;
  if (f < 0 || f > 1)
    return false;	// Segment does not intersect triangle plane.

  // Point in triangle plane relative to va.
  float pp[3] = {p[0]+f*t[0], p[1]+f*t[1], p[2]+f*t[2]};

  // Check if plane intersection point lies within triangle.

  float nba[3] = {n[1]*ba[2]-n[2]*ba[1],	// Normal to edge ba.
		  n[2]*ba[0]-n[0]*ba[2],
		  n[0]*ba[1]-n[1]*ba[0]};
  float bc = nba[0]*ca[0] + nba[1]*ca[1] + nba[2]*ca[2];
  float pc = nba[0]*pp[0] + nba[1]*pp[1] + nba[2]*pp[2];
  if (pc < 0 || pc > bc)
    return false;	// ca coefficient is not in [0,1].

  float nca[3] = {ca[1]*n[2]-ca[2]*n[1],	// Normal to edge ca
		  ca[2]*n[0]-ca[0]*n[2],
		  ca[0]*n[1]-ca[1]*n[0]};
  float pb = nca[0]*pp[0] + nca[1]*pp[1] + nca[2]*pp[2];
  if (pb < 0 || pb > bc)
    return false;	// ba coefficient is not in [0,1].

  if (pc + pb > bc)
    return false;	// Wrong side of edge bc

  *fret = f;
  return true;
}

const char *closest_sphere_intercept_doc =
  "closest_sphere_intercept(centers, radii, xyz1, xyz2) -> f, snum\n"
  "\n"
  "Find first sphere intercept along line segment from xyz1 to xyz2.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "centers : n by 3 float array\n"
  "  x,y,z sphere center coordinates.\n"
  "radii : length n float array\n"
  "  sphere radii.\n"
  "xyz1, xyz2 : float 3-tuples\n"
  "  x,y,z coordinates of line segment endpoints.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "f : float\n"
  "  fraction of distance from xyz1 to xyz2.  None if no intercept.\n"
  "snum : int\n"
  "  sphere number, or None if no intercept.\n";

// ----------------------------------------------------------------------------
// Find closest sphere point intersectiong the line segment between xyz1 and xyz2.
// Returns fraction of way along segment and the sphere number.
//
extern "C"
PyObject *closest_sphere_intercept(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray centers, radii;
  float xyz1[3], xyz2[3];
  const char *kwlist[] = {"centers", "radii", "xyz1", "xyz2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &radii,
				   parse_float_3_array, &xyz1,
				   parse_float_3_array, &xyz2))
    return NULL;

  if (radii.size(0) != centers.size(0))
    {
      PyErr_SetString(PyExc_ValueError,
		      "closest_sphere_intercept(): radii and center arrays must have same size");
      return NULL;
    }

  float fmin;
  int s;
  PyObject *py_fmin, *py_snum;
  if (closest_sphere_intercept(centers.values(), centers.size(0), centers.stride(0), centers.stride(1),
			       radii.values(), radii.stride(0),
			       xyz1, xyz2, &fmin, &s))
    {
      py_fmin = PyFloat_FromDouble(fmin);
      py_snum = PyLong_FromLong(s);
    }
  else
    {
      py_fmin = Py_None; Py_INCREF(Py_None);
      py_snum = Py_None; Py_INCREF(Py_None);
    }
  PyObject *t = python_tuple(py_fmin, py_snum);

  return t;
}

// ----------------------------------------------------------------------------
//
static bool closest_sphere_intercept(const float *centers, int n, int cstride0, int cstride1,
				     const float *radii, int rstride,
				     const float *xyz1, const float *xyz2,
				     float *fmin, int *snum)
{
  float x1 = xyz1[0], y1 = xyz1[1], z1 = xyz1[2];
  float dx = xyz2[0]-xyz1[0], dy = xyz2[1]-xyz1[1], dz = xyz2[2]-xyz1[2];
  float d = sqrt(dx*dx + dy*dy + dz*dz);
  if (d == 0)
    return false;
  dx /= d; dy /= d; dz /= d;

  float dc = 2*d;
  int sc;
  for (int s = 0 ; s < n ; ++s)
    {
      int s3 = cstride0*s;
      float x = centers[s3], y = centers[s3+cstride1], z = centers[s3+2*cstride1], r = radii[s*rstride];
      float p = (x-x1)*dx + (y-y1)*dy + (z-z1)*dz;
      if (p >= 0 && p <= d + r && p < dc + r)
	{
	  float xp = x-(x1+p*dx), yp = y-(y1+p*dy), zp = z-(z1+p*dz);	// perp vector
	  float d2 = xp*xp + yp*yp + zp*zp;	// perp distance squared
	  float a2 = r*r - d2;
	  if (a2 > 0)
	    {
	      float a = sqrt(a2);
	      if (p-a < dc)
		{
		  dc = p-a;
		  sc = s;
		}
	    }
	}
    }
  if (dc > d)
    return false;

  *fmin = dc/d;
  *snum = sc;
  return true;
}

const char *closest_cylinder_intercept_doc =
  "closest_cylinder_intercept(base1, base2, radii, xyz1, xyz2) -> f, cnum\n"
  "\n"
  "Find first cylinder intercept along line segment from xyz1 to xyz2.\n"
  "Cylinder endcaps are not considered. Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "base1, base2 : n by 3 float arrays\n"
  "  x,y,z cylinder end center coordinates.\n"
  "radii : float array\n"
  "  cylinder radii.\n"
  "xyz1, xyz2 : float 3-tuples\n"
  "  x,y,z coordinates of line segment endpoints.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "f : float\n"
  "  fraction of distance from xyz1 to xyz2.  None if no intercept.\n"
  "cnum : int\n"
  "  cylinder number, or None if no intercept.\n";

// ----------------------------------------------------------------------------
// Find closest cylinder point intersectiong the line segment between xyz1 and xyz2.
// Intercept with cylinder end caps is not considered.
// Returns fraction of way along segment and the cylinder number.
//
extern "C"
PyObject *closest_cylinder_intercept(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray base1, base2, radii;
  float xyz1[3], xyz2[3];
  const char *kwlist[] = {"base1", "base2", "radii", "xyz1", "xyz2", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &base1,
				   parse_float_n3_array, &base2,
				   parse_float_n_array, &radii,
				   parse_float_3_array, &xyz1,
				   parse_float_3_array, &xyz2))
    return NULL;

  if (radii.size(0) != base1.size(0) || base2.size(0) != base1.size(0))
    {
      PyErr_SetString(PyExc_ValueError,
		      "closest_cylinder_intercept(): radii, base1, and base2 arrays must have same size");
      return NULL;
    }

  float fmin;
  int c;
  PyObject *py_fmin, *py_cnum;
  if (closest_cylinder_intercept(base1.values(), base1.size(0), base1.stride(0), base1.stride(1),
				 base2.values(), base1.stride(0), base1.stride(1),
				 radii.values(), radii.stride(0),
				 xyz1, xyz2, &fmin, &c))
    {
      py_fmin = PyFloat_FromDouble(fmin);
      py_cnum = PyLong_FromLong(c);
    }
  else
    {
      py_fmin = Py_None; Py_INCREF(Py_None);
      py_cnum = Py_None; Py_INCREF(Py_None);
    }
  PyObject *t = python_tuple(py_fmin, py_cnum);

  return t;
}

// ----------------------------------------------------------------------------
//
static bool closest_cylinder_intercept(const float *base1, int n, int b1stride0, int b1stride1,
				       const float *base2, int b2stride0, int b2stride1,
				       const float *radii, int rstride,
				       const float *xyz1, const float *xyz2,
				       float *fmin, int *cnum)
{
  float cx = xyz1[0], cy = xyz1[1], cz = xyz1[2];
  float dx = xyz2[0]-xyz1[0], dy = xyz2[1]-xyz1[1], dz = xyz2[2]-xyz1[2];
  float d2 = dx*dx + dy*dy + dz*dz;
  if (d2 == 0)
    return false;

  // Optimization to quickly detect if infinite line misses infinite cylinder.
  float px = -dz, pz = dx, pn = sqrtf(dx*dx + dz*dz);
  if (pn > 0)
    { px /= pn; pz /= pn; }
  float pc = -(px*cx + pz*cz);

  float fc = 2;
  int cc;
  for (int c = 0 ; c < n ; ++c)
    {
      int s3 = b1stride0*c;
      float bx = base1[s3], by = base1[s3+b1stride1], bz = base1[s3+2*b1stride1], r = radii[c*rstride];
      s3 = b2stride0*c;
      float b2x = base2[s3], b2y = base2[s3+b2stride1], b2z = base2[s3+2*b2stride1];

      // Optimization to quickly detect if infinite line misses infinite cylinder.
      float w1 = px*bx + pz*bz + pc, w2 = px*b2x + pz*b2z + pc;
      if ((w1 > r && w2 > r) || (w1 < -r && w2 < -r))
	continue;

      float ax = b2x-bx, ay = b2y-by, az = b2z-bz;
      float a2 = ax*ax + ay*ay + az*az;
      float ad = ax*dx + ay*dy + az*dz;
      float ad2 = ad*ad;
      float ex = cx-bx, ey = cy-by, ez = cz-bz;
      float e2 = ex*ex + ey*ey + ez*ez;
      float ed = ex*dx + ey*dy + ez*dz;
      float ea = ex*ax + ey*ay + ez*az;
      float r2 = r*r;

      float q2 = a2*d2 - ad2, q1 = ed*a2 - ea*ad, q0 = e2*a2 - r2*a2 - ea*ea;
      if (q2 <= 0)
	continue;	// No intercept, line parallel to cylinder

      float qd = q1*q1 - q2*q0;
      if (qd < 0)
	continue;	// No intercept, cylinder radius too small.

      float q = sqrtf(qd);
      float f1 = (-q1 - q) / q2, f2 = (-q1 + q) / q2;
      float p1 = ea + f1*ad, p2 = ea + f2*ad;
      float f;
      if (f1 >= 0 && f1 <= 1 && p1 >= 0 && p1 <= a2)
	f = f1;
      else if (f2 >= 0 && f2 <= 1 && p2 >= 0 && p2 <= a2)
	f = f2;
      else
	continue;	// Intercepts beyond ends of line segment or beyond ends of cylinder

      if (f >= fc)
	continue;	// Already have a closer cylinder

      fc = f;
      cc = c;
    }

  if (fc > 1)
    return false;

  *fmin = fc;
  *cnum = cc;
  return true;
}
