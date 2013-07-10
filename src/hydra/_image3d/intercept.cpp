// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

static bool closest_geometry_intercept(const float *varray,
				       const int *tarray, int nt,
				       const float *xyz1, const float *xyz2,
				       float *fmin, int *tmin);
static bool triangle_intercept(const float *va, const float *vb,
			       const float *vc,
			       const float *xyz1, const float *xyz2,
			       float *fret);
static bool closest_sphere_intercept(const float *centers, const float *radii, int n,
				     const float *xyz1, const float *xyz2,
				     float *fmin, int *snum);

// ----------------------------------------------------------------------------
// Find closest triangle intercepting line segment between xyz1 and xyz2.
// The vertex array is xyz points (n by 3, NumPy single).
// The triangle array is triples of indices into the vertex array (m by 3,
// NumPy intc).
// Returns fraction of way along segment triangle index.
//
extern "C"
PyObject *closest_geometry_intercept(PyObject *s, PyObject *args, PyObject *keywds)
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
  if (closest_geometry_intercept(vertices.values(), triangles.values(), triangles.size(0),
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
static bool closest_geometry_intercept(const float *varray,
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
  if (closest_sphere_intercept(centers.values(), radii.values(), centers.size(0),
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
static bool closest_sphere_intercept(const float *centers, const float *radii, int n,
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
      int s3 = 3*s;
      float x = centers[s3], y = centers[s3+1], z = centers[s3+2], r = radii[s];
      float p = (x-x1)*dx + (y-y1)*dy + (z-z1)*dz;
      if (p >= 0 && p <= d && p < dc)
	{
	  float xp = x-(x1+p*dx), yp = y-(y1+p*dy), zp = z-(z1+p*dz);	// perp vector
	  float d2 = xp*xp + yp*yp + zp*zp;	// perp distance squared
	  if (d2 < r*r)
	    {
	      dc = p;
	      sc = s;
	    }
	}
    }
  if (dc > d)
    return false;

  *fmin = dc/d;
  *snum = sc;
  return true;
}
