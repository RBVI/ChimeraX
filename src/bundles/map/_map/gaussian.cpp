// vi: set expandtab shiftwidth=4 softtabstop=4:

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

// ----------------------------------------------------------------------------
// Add Gaussians of specified center, amplitude and width to a 3-d floating
// point array.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use ceil(), floor(), exp()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray

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
  const int64_t *msize = matrix.sizes();
  int64_t n = centers.size(0);
  const float *ca = centers.values();
  int64_t cs0 = centers.stride(0), cs1 = centers.stride(1);
  const float *cfa = coef.values(), *sa = sdev.values();
  int64_t cfs0 = coef.stride(0), ss0 = sdev.stride(0), ss1 = sdev.stride(1);
  float *ma = matrix.values();
  int64_t ms0 = matrix.stride(0), ms1 = matrix.stride(1), ms2 = matrix.stride(2);
  for (int64_t c = 0 ; c < n ; ++c)
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

  Py_BEGIN_ALLOW_THREADS
  sum_of_gaussians(centers, coef, sdev, maxrange, matrix);
  Py_END_ALLOW_THREADS

  return python_none();
}

// -----------------------------------------------------------------------------
//
static void sum_of_balls(const FArray &centers, const FArray &radii,
			 float sdev, float maxrange, FArray &matrix)
{
  const int64_t *msize = matrix.sizes();
  int64_t n = centers.size(0);
  const float *ca = centers.values();
  int64_t cs0 = centers.stride(0), cs1 = centers.stride(1);
  const float *ra = radii.values();
  int64_t rs0 = radii.stride(0);
  float *ma = matrix.values();
  int64_t ms0 = matrix.stride(0), ms1 = matrix.stride(1), ms2 = matrix.stride(2);
  for (int64_t c = 0 ; c < n ; ++c)
    {
      float cijk[3];
      int ijk_min[3], ijk_max[3];
      float r = ra[rs0*c];
      float r2 = r*r;
      for (int p = 0 ; p < 3 ; ++p)
	{
	  float x = ca[cs0*c+cs1*p];
	  cijk[p] = x;
	  ijk_min[p] = clamp((int)ceil(x-r-maxrange*sdev), msize[2-p]);
	  ijk_max[p] = clamp((int)floor(x+r+maxrange*sdev), msize[2-p]);
	}
      for (int k = ijk_min[2] ; k <= ijk_max[2] ; ++k)
	{
	  float dk = (k-cijk[2]);
	  float k2 = dk*dk;
	  for (int j = ijk_min[1] ; j <= ijk_max[1] ; ++j)
	    {
	      float dj = (j-cijk[1]);
	      float jk2 = dj*dj + k2;
	      for (int i = ijk_min[0] ; i <= ijk_max[0] ; ++i)
		{
		  float di = (i-cijk[0]);
		  float ijk2 = di*di + jk2;
		  float v = 1;
		  if (ijk2 > r2)
		    {
		      float gr = (sqrt(ijk2) - r)/sdev;
		      v = exp(-0.5*gr*gr);
		    }
		  ma[k*ms0+j*ms1+i*ms2] += v;
		}
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_sum_of_balls(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray centers, radii, matrix;
  float sdev, maxrange;
  const char *kwlist[] = {"centers", "radii", "sdev", "maxrange", "matrix", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&ffO&"), (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &radii,
				   &sdev, &maxrange,
				   parse_writable_float_3d_array, &matrix))
    return NULL;

  if (radii.size(0) != centers.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "Lengths of centers and radii don't match.");
      return NULL;
    }

  Py_BEGIN_ALLOW_THREADS
  sum_of_balls(centers, radii, sdev, maxrange, matrix);
  Py_END_ALLOW_THREADS

  return python_none();
}

// ----------------------------------------------------------------------------
//
static void covariance_sum(float c[3][3], float center[3], float scale, FArray &array)
{
  int ksize = array.size(0), jsize = array.size(1), isize = array.size(2);
  float i0 = center[0], j0 = center[1], k0 = center[2];
  int64_t as0 = array.stride(0), as1 = array.stride(1), as2 = array.stride(2);
  float *aa = array.values();
  for (int k = 0 ; k < ksize ; ++k)
    for (int j = 0 ; j < jsize ; ++j)
      for (int i = 0 ; i < isize ; ++i)
	{
	  float vi = i-i0, vj = j-j0, vk = k-k0;
	  float e = (vi * (c[0][0]*vi + c[0][1]*vj + c[0][2]*vk) +
		     vj * (c[1][0]*vi + c[1][1]*vj + c[1][2]*vk) +
		     vk * (c[2][0]*vi + c[2][1]*vj + c[2][2]*vk));
	  aa[as0*k + as1*j + as2*i] += scale*expf(-0.5*e);
	}
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *covariance_sum(PyObject *, PyObject *args, PyObject *keywds)
{
  float cov_inv[3][3], center[3], scale;
  FArray array;
  const char *kwlist[] = {"cov_inv", "center", "scale", "array", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&fO&"), (char **)kwlist,
				   parse_float_3x3_array, &cov_inv[0][0],
				   parse_float_3_array, &center,
				   &scale,
				   parse_writable_float_3d_array, &array))
    return NULL;

  Py_BEGIN_ALLOW_THREADS
  covariance_sum(cov_inv, center, scale, array);
  Py_END_ALLOW_THREADS

  return python_none();
}
