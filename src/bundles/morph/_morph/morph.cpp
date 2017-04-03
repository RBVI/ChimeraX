/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <Python.h>			// use PyObject
#include <math.h>	// use sqrt()
#include <arrays/pythonarray.h>		// use parse_double_3_array()

#ifndef M_PI
// M_PI is not part of ANSI C and Windows does not define it
#define M_PI 3.14159265358979323846
#endif

static double distance(double *u, double *v)
{
  double x = u[0]-v[0], y = u[1]-v[1], z = u[2]-v[2];
  return sqrt(x*x + y*y + z*z);
}

static double inner_product(double *u, double *v)
{
  double ip = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
  return ip;
}

static double norm(double *u)
{
  double ip = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
  return sqrt(ip);
}

static void normalize_vector(double *u)
{
  double n = norm(u);
  if (n > 0)
    {
      u[0] /= n;
      u[1] /= n;
      u[2] /= n;
    }
}

static void subtract(double *u, double *v, double *uv)
{
  uv[0] = u[0]-v[0];
  uv[1] = u[1]-v[1];
  uv[2] = u[2]-v[2];
}

static void cross_product(double *u, double *v, double *uv)
{
  uv[0] = u[1]*v[2]-u[2]*v[1];
  uv[1] = u[2]*v[0]-u[0]*v[2];
  uv[2] = u[0]*v[1]-u[1]*v[0];
}

static double degrees(double radians)
{
  return radians * 180.0 / M_PI;
}

static double radians(double degrees)
{
  return M_PI * degrees / 180.0;
}

static double angle(double *p0, double *p1, double *p2)
{
  double v0[3], v1[3];
  subtract(p0,p1,v0);
  subtract(p2,p1,v1);
  double acc = inner_product(v0, v1);
  double d0 = norm(v0);
  double d1 = norm(v1);
  if (d0 <= 0 || d1 <= 0)
    return 0;
  acc /= (d0 * d1);
  if (acc > 1)
    acc = 1;
  else if (acc < -1)
    acc = -1;
  return degrees(acos(acc));
}

static double angle(double *v0, double *v1)
{
  double acc = inner_product(v0, v1);
  double d0 = norm(v0);
  double d1 = norm(v1);
  if (d0 <= 0 || d1 <= 0)
    return 0;
  acc /= (d0 * d1);
  if (acc > 1)
    acc = 1;
  else if (acc < -1)
    acc = -1;
  return degrees(acos(acc));
}

static double dihedral(double *p0, double *p1, double *p2, double *p3)
{
  double v10[3], v12[3], v23[3], t[3], u[3], v[3];
  subtract(p1, p0, v10);
  subtract(p1, p2, v12);
  subtract(p2, p3, v23);
  cross_product(v10, v12, t);
  cross_product(v23, v12, u);
  cross_product(u, t, v);
  double w = inner_product(v, v12);
  double acc = angle(u, t);
  if (w < 0)
    acc = -acc;
  return acc;
}

static void dihedral_point(double *n1, double *n2, double *n3, double dist, double angle, double dihed,
			   double *dp)
{
  // Find dihedral point n0 with specified n0 to n1 distance,
  // n0,n1,n2 angle, and n0,n1,n2,n3 dihedral (angles in degrees).

  double v12[3], v13[3], x[3], y[3];
  subtract(n2, n1, v12);
  subtract(n3, n1, v13);
  normalize_vector(v12);
  cross_product(v13, v12, x);
  normalize_vector(x);
  cross_product(v12, x, y);
  normalize_vector(y);

  double radAngle = radians(angle);
  double tmp = dist * sin(radAngle);
  double radDihed = radians(dihed);
  double xc = tmp*sin(radDihed), yc = tmp*cos(radDihed), zc = dist*cos(radAngle);
  for (int a = 0 ; a < 3 ; ++a)
    dp[a] = xc*x[a] + yc*y[a] + zc*v12[a] + n1[a];
}

// Every 4 indices in array i define a dihedral.
static void interp_dihedrals(const IArray &i, const DArray &coords0, const DArray &coords1,
			     double f, DArray &coords)
{
  long n = i.size();
  int *ia = i.values();
  double *ca0 = coords0.values(), *ca1 = coords1.values(), *ca = coords.values();
  for (long j = 0 ; j < n ; j += 4)
    {
      int i0 = 3*ia[j], i1 = 3*ia[j+1], i2 = 3*ia[j+2], i3 = 3*ia[j+3];
      double *c00 = &ca0[i0], *c01 = &ca0[i1], *c02 = &ca0[i2], *c03 = &ca0[i3];
      double *c10 = &ca1[i0], *c11 = &ca1[i1], *c12 = &ca1[i2], *c13 = &ca1[i3];
      double *c0 = &ca[i0], *c1 = &ca[i1], *c2 = &ca[i2], *c3 = &ca[i3];
      double length0 = distance(c00, c01);
      double angle0 = angle(c00, c01, c02);
      double dihed0 = dihedral(c00, c01, c02, c03);
      double length1 = distance(c10, c11);
      double angle1 = angle(c10, c11, c12);
      double dihed1 = dihedral(c10, c11, c12, c13);
      double length = length0 + (length1 - length0) * f;
      double angle = angle0 + (angle1 - angle0) * f;
      double ddihed = dihed1 - dihed0;
      if (ddihed > 180)
	ddihed -= 360;
      else if (ddihed < -180)
	ddihed += 360;
      double dihed = dihed0 + ddihed * f;
      dihedral_point(c1, c2, c3, length, angle, dihed, c0);
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
interpolate_dihedrals(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray i;
  double f;
  DArray coords0, coords1, coords;
  const char *kwlist[] = {"i", "coords0", "coords1", "f", "coords", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&dO&"),
				   (char **)kwlist,
                                   parse_int_n_array, &i,
				   parse_double_n3_array, &coords0,
				   parse_double_n3_array, &coords1,
				   &f,
				   parse_writable_double_n3_array, &coords))
    return NULL;
  if (!i.is_contiguous() || !coords0.is_contiguous() || !coords.is_contiguous() || !coords.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Arrays must be contiguous");
      return NULL;
    }

  interp_dihedrals(i, coords0, coords1, f, coords);
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
static void interp_linear(const IArray &i, const DArray &coords0, const DArray &coords1,
			  double f, DArray &coords)
{
  long n = i.size();
  int *ia = i.values();
  double f0 = 1-f;
  double *ca0 = coords0.values(), *ca1 = coords1.values(), *ca = coords.values();
  for (long j = 0 ; j < n ; ++j)
    {
      int i3 = 3*ia[j];
      double *ca0i = ca0+i3, *ca1i = ca1+i3, *cai = ca+i3;
      for (int a = 0 ; a < 3 ; ++a)
	cai[a] = ca0i[a]*f0 + ca1i[a]*f;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
interpolate_linear(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray i;
  double f;
  DArray coords0, coords1, coords;
  const char *kwlist[] = {"i", "coords0", "coords1", "f", "coords", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&dO&"),
				   (char **)kwlist,
                                   parse_int_n_array, &i,
				   parse_double_n3_array, &coords0,
				   parse_double_n3_array, &coords1,
				   &f,
				   parse_writable_double_n3_array, &coords))
    return NULL;
  if (!i.is_contiguous() || !coords0.is_contiguous() || !coords.is_contiguous() || !coords.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Arrays must be contiguous");
      return NULL;
    }

  interp_linear(i, coords0, coords1, f, coords);
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
static void  rigid_motion(DArray &coords, const IArray &i,
                          double *axis, double angle, double *center, double *shift, double f)
{
  double arad = f * angle * M_PI / 180.0;
  double sa = sin(arad), ca = cos(arad);
  double k = 1 - ca;
  double ax = axis[0], ay = axis[1], az = axis[2];
  double r00 = 1 + k * (ax * ax - 1);
  double r01 = -az * sa + k * ax * ay;
  double r02 = ay * sa + k * ax * az;
  double r10 = az * sa + k * ax * ay;
  double r11 = 1 + k * (ay * ay - 1);
  double r12 = -ax * sa + k * ay * az;
  double r20 = -ay * sa + k * ax * az;
  double r21 = ax * sa + k * ay * az;
  double r22 = 1 + k * (az * az - 1);
  double cx = center[0], cy = center[1], cz = center[2];
  double rcx = r00*cx + r01*cy + r02*cz;
  double rcy = r10*cx + r11*cy + r12*cz;
  double rcz = r20*cx + r21*cy + r22*cz;
  double sx = cx - rcx + f*shift[0], sy = cy - rcy + f*shift[1], sz = cz - rcz + f*shift[2];

  long n = i.size();
  int *ia = i.values();
  double *crd = coords.values();
  for (long j = 0 ; j < n ; ++j)
    {
      double *ci = crd + 3*ia[j];
      double x = ci[0], y = ci[1], z = ci[2];
      ci[0] = r00*x + r01*y + r02*z + sx;
      ci[1] = r10*x + r11*y + r12*z + sy;
      ci[2] = r20*x + r21*y + r22*z + sz;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
apply_rigid_motion(PyObject *, PyObject *args, PyObject *keywds)
{
  DArray coords;
  IArray i;
  double axis[3], angle, center[3], shift[3], f;
  const char *kwlist[] = {"coords", "i", "axis", "angle", "center", "shift", "f", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&dO&O&d"),
				   (char **)kwlist,
				   parse_writable_double_n3_array, &coords,
                                   parse_int_n_array, &i,
				   parse_double_3_array, &axis[0],
                                   &angle,
				   parse_double_3_array, &center[0],
				   parse_double_3_array, &shift[0],
				   &f))
    return NULL;
  if (!i.is_contiguous() || !coords.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "Arrays must be contiguous");
      return NULL;
    }

  rigid_motion(coords, i, axis, angle, center, shift, f);
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
static PyMethodDef morph_methods[] = {
  {const_cast<char*>("interpolate_dihedrals"), (PyCFunction)interpolate_dihedrals,
   METH_VARARGS|METH_KEYWORDS,
   "interplate_dihedrals(...)\n"
   "\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("interpolate_linear"), (PyCFunction)interpolate_linear,
   METH_VARARGS|METH_KEYWORDS,
   "interplate_linear(...)\n"
   "\n"
   "Implemented in C++.\n"
  },
  {const_cast<char*>("apply_rigid_motion"), (PyCFunction)apply_rigid_motion,
   METH_VARARGS|METH_KEYWORDS,
   "apply_rigid_motion(...)\n"
   "\n"
   "Implemented in C++.\n"
  },
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef morph_def =
{
	PyModuleDef_HEAD_INIT,
	"_morph",
	"Morph utility routines",
	-1,
	morph_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__morph()
{
	return PyModule_Create(&morph_def);
}
