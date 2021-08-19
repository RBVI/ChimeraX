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
// Compute signed distance from surface of union of spheres.  This is used
// for computing solvent accessible and solvent excluded molecular surfaces
// as grid isosurfaces.
//
#include <Python.h>			// use PyObject
#include <math.h>			// use ceil(), floor(), sqrt()

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
static void sphere_surface_distance(const FArray &centers, const FArray &radii,
				    float maxrange, FArray &matrix)
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
      float r = ra[c*rs0];
      if (r == 0)
	continue;
      float cijk[3];
      int ijk_min[3], ijk_max[3];
      for (int p = 0 ; p < 3 ; ++p)
	{
	  float x = ca[cs0*c+cs1*p];
	  cijk[p] = x;
	  ijk_min[p] = clamp((int)ceil(x-(r+maxrange)), msize[2-p]);
	  ijk_max[p] = clamp((int)floor(x+(r+maxrange)), msize[2-p]);
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
		  float rd = sqrt(ijk2) - r;
		  float *mijk = ma + (k*ms0+j*ms1+i*ms2);
		  if (rd < *mijk)
		    *mijk = rd;
		}
	    }
	}
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_sphere_surface_distance(PyObject *, PyObject *args,
						PyObject *keywds)
{
  FArray centers, radii, matrix;
  float maxrange;
  const char *kwlist[] = {"centers", "radii", "maxrange", "matrix", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&fO&"), (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n_array, &radii,
				   &maxrange,
				   parse_writable_float_3d_array, &matrix))
    return NULL;

  if (radii.size(0) != centers.size(0))
    {
      PyErr_SetString(PyExc_TypeError,
		      "Lengths of centers and radii arrays don't match.");
      return NULL;
    }

  Py_BEGIN_ALLOW_THREADS
  sphere_surface_distance(centers, radii, maxrange, matrix);
  Py_END_ALLOW_THREADS

  return python_none();
}
