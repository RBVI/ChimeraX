#include <Python.h>			// use PyObject
#include <math.h>	// use sqrt()
#include <arrays/pythonarray.h>		// use parse_double_3_array()

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

static void interp_dihedral(double *c00, double *c01, double *c02, double *c03,
			    double *c10, double *c11, double *c12, double *c13,
			    double f,
			    double *c1, double *c2, double *c3, double *c0)
{
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

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
interpolate_dihedral(PyObject *, PyObject *args, PyObject *keywds)
{
  double c00[3], c01[3], c02[3], c03[3], c10[3], c11[3], c12[3], c13[3], f, c1[3], c2[3], c3[3], c0[3];
  const char *kwlist[] = {"c00", "c01", "c02", "c03", "c10", "c11", "c12", "c13", "f",
			  "c1", "c2", "c3", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&O&O&O&O&dO&O&O&"),
				   (char **)kwlist,
				   parse_double_3_array, &c00[0],
				   parse_double_3_array, &c01[0],
				   parse_double_3_array, &c02[0],
				   parse_double_3_array, &c03[0],
				   parse_double_3_array, &c10[0],
				   parse_double_3_array, &c11[0],
				   parse_double_3_array, &c12[0],
				   parse_double_3_array, &c13[0],
				   &f,
				   parse_double_3_array, &c1[0],
				   parse_double_3_array, &c2[0],
				   parse_double_3_array, &c3[0]))
    return NULL;

  interp_dihedral(c00, c01, c02, c03, c10, c11, c12, c13, f, c1, c2, c3, c0);
  
  return c_array_to_python(&c0[0], 3);
}

// ----------------------------------------------------------------------------
//
static PyMethodDef morph_methods[] = {
  {const_cast<char*>("interpolate_dihedral"), (PyCFunction)interpolate_dihedral,
   METH_VARARGS|METH_KEYWORDS,
   "interplate_dihedral(...)\n"
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
