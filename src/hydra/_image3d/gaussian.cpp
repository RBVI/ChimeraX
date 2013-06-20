// ----------------------------------------------------------------------------
// Add Gaussians of specified center, amplitude and width to a 3-d floating
// point array.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use ceil(), floor(), exp()

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray

// -----------------------------------------------------------------------------
//
static inline int clamp(int x, int limit)
{
  if (x < 0)
    return 0;
  else if (x >= limit)
    return limit-1;
  return x;
}

// -----------------------------------------------------------------------------
//
static void sum_of_gaussians(const FArray &centers, const FArray &coef,
			     const FArray &sdev, float maxrange,
			     FArray &matrix)
{
  const int *msize = matrix.sizes();
  int n = centers.size(0);
  const float *ca = centers.values();
  int cs0 = centers.stride(0), cs1 = centers.stride(1);
  const float *cfa = coef.values(), *sa = sdev.values();
  int cfs0 = coef.stride(0), ss0 = sdev.stride(0), ss1 = sdev.stride(1);
  float *ma = matrix.values();
  int ms0 = matrix.stride(0), ms1 = matrix.stride(1), ms2 = matrix.stride(2);
  for (int c = 0 ; c < n ; ++c)
    {
      float sd[3] = {sa[c*ss0], sa[c*ss0 + ss1], sa[c*ss0 + 2*ss1]};
      if (sd[0] == 0 || sd[1] == 0 || sd[2] == 0)
	continue;
      float cijk[3];
      int ijk_min[3], ijk_max[3];
      for (int p = 0 ; p < 3 ; ++p)
	{
	  float x = ca[cs0*c+cs1*p];
	  cijk[p] = x;
	  ijk_min[p] = clamp((int)ceil(x-maxrange*sd[p]), msize[2-p]);
	  ijk_max[p] = clamp((int)floor(x+maxrange*sd[p]), msize[2-p]);
	}
      float cf = cfa[c*cfs0];
      for (int k = ijk_min[2] ; k <= ijk_max[2] ; ++k)
	{
	  float dk = (k-cijk[2])/sd[2];
	  float k2 = dk*dk;
	  for (int j = ijk_min[1] ; j <= ijk_max[1] ; ++j)
	    {
	      float dj = (j-cijk[1])/sd[1];
	      float jk2 = dj*dj + k2;
	      for (int i = ijk_min[0] ; i <= ijk_max[0] ; ++i)
		{
		  float di = (i-cijk[0])/sd[0];
		  float ijk2 = di*di + jk2;
		  ma[k*ms0+j*ms1+i*ms2] += cf*exp(-0.5*ijk2);
		}
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_sum_of_gaussians(PyObject *, PyObject *args,
					 PyObject *keywds)
{
  FArray centers, coef, sdev, matrix;
  float maxrange;
  const char *kwlist[] = {"centers", "coef", "sdev", "maxrange", "matrix", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&fO&"), (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &coef,
				   parse_float_n3_array, &sdev,
				   &maxrange,
				   parse_writable_float_3d_array, &matrix))
    return NULL;

  if (coef.size(0) != centers.size(0) || sdev.size(0) != centers.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "Lengths of centers, coef, sdev arrays don't match.");
      return NULL;
    }

  sum_of_gaussians(centers, coef, sdev, maxrange, matrix);

  Py_INCREF(Py_None);
  return Py_None;
}
