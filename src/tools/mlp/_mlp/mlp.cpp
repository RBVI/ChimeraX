// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Compute lipophilicity by summing atom values with a distance function
// over a 3d grid.
//
#include <Python.h>			// use PyObject

//#include <iostream>			// use std::cerr for debugging

#include <math.h>			// use sqrtf(), expf()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray, IArray

inline int max(int a, int b) { return (a < b ? b : a); }
inline int min(int a, int b) { return (a < b ? a : b); }

// ----------------------------------------------------------------------------
//
static void lipophilicity_sum(const FArray &xyz, const FArray &fi,
			      float origin[3], float spacing, float max_dist, float nexp,
			      const char *method,  FArray &pot)
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  int na = xyz.size(0);
  long xs0 = xyz.stride(0), xs1 = xyz.stride(1);
  const float *xyza = xyz.values();
  int nz = pot.size(0), ny = pot.size(1), nx = pot.size(2);
  long ps0 = pot.stride(0), ps1 = pot.stride(1), ps2 = pot.stride(2);
  float *pa = pot.values();
  long fs0 = fi.stride(0);
  const float *fa = fi.values();
  int md = int(ceil(max_dist / spacing));
  for (int a = 0 ; a < na ; ++a)
    {
      const float *xa = xyza + xs0*a;
      float ax = *xa, ay = xa[xs1], az = xa[2*xs1];
      float f = 100 * fa[fs0*a];
      float i0 = (ax-x0)/spacing, j0 = (ay-y0)/spacing, k0 = (az-z0)/spacing;
      int kmin = max(0,int(floor(k0-md))), kmax = min(nz-1,int(ceil(k0+md)));
      int jmin = max(0,int(floor(j0-md))), jmax = min(ny-1,int(ceil(j0+md)));
      int imin = max(0,int(floor(i0-md))), imax = min(nx-1,int(ceil(i0+md)));
      for (int k = kmin ; k <= kmax ; ++k)
	{
	  float gz = z0 + k * spacing;
	  for (int j = jmin ; j <= jmax ; ++j)
	    {
	      float gy = y0 + j * spacing;
	      for (int i = imin ; i <= imax ; ++i)
		{
		  // Evaluation of the distance between the grid point and each atoms
		  float gx = x0 + i * spacing;
		  float dx = ax-gx, dy = ay-gy, dz = az-gz;
		  float d = sqrtf(dx*dx + dy*dy + dz*dz);
		  if (d <= max_dist)
		    pa[ps0*k+ps1*j+ps2*i] += f * expf(-d);	// Fauchere
		}
	    }
	}
    }
}

/*
def _dubost(fi, d, n):
    return (100 * fi / (1 + d)).sum()

def _fauchere(fi, d, n):
    from numpy import exp
    return (100 * fi * exp(-d)).sum()

def _brasseur(fi, d, n):
    #3.1 division is there to remove any units in the equation
    #3.1A is the average diameter of a water molecule (2.82 -> 3.2)
    from numpy import exp
    return (100 * fi * exp(-d/3.1)).sum()

def _buckingham(fi, d, n):
    return (100 * fi / (d**n)).sum()

def _type5(fi, d, n):
    from numpy import exp, sqrt
    return (100 * fi * exp(-sqrt(d))).sum()
*/

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
mlp_sum(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray xyz, fi, pot;
  float origin[3], spacing, max_dist, nexp;
  const char *method;
  const char *kwlist[] = {"xyz", "fi", "origin", "spacing", "max_dist", "method", "nexp", "pot", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&ffsfO&"),
				   (char **)kwlist,
				   parse_float_n3_array, &xyz,
				   parse_float_n_array, &fi,
				   parse_float_3_array, &origin[0],
				   &spacing,
				   &max_dist,
				   &method,
				   &nexp,
				   parse_writable_float_3d_array, &pot))
    return NULL;

  if (xyz.size(0) != fi.size(0))
    return PyErr_Format(PyExc_ValueError, "Xyz and fi arrays have different sizes %d and %d",
			xyz.size(0), fi.size(0));


  lipophilicity_sum(xyz, fi, origin, spacing, max_dist, nexp, method, pot);
  
  return python_none();
}

// ----------------------------------------------------------------------------
//
static PyMethodDef mlp_methods[] = {
  {const_cast<char*>("mlp_sum"), (PyCFunction)mlp_sum,
   METH_VARARGS|METH_KEYWORDS,
   "mlp_sum(xyz, fi, origin, spacing, method, nexp, pot)\n"
   "\n"
   "Sum lipophilicity values for atoms over a grid.\n"
   "Implemented in C++.\n"
  },
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mlp_def =
{
	PyModuleDef_HEAD_INIT,
	"_mlp",
	"Lipophilicity calculation",
	-1,
	mlp_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__mlp()
{
	return PyModule_Create(&mlp_def);
}
