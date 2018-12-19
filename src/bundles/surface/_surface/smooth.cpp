// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>	// use parse_float_n3_array(), ...
#include <arrays/rcarray.h>	// use IArray, FArray

static void smooth_vertices(FArray &varray, const IArray &tarray,
			    float smoothing_factor, int smoothing_iterations);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *smooth_vertex_positions(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray varray;
  IArray tarray;
  float smoothing_factor;
  int smoothing_iterations;
  const char *kwlist[] = {"vertices", "triangles", "smoothing_factor", "smoothing_iterations", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&fi"), (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   &smoothing_factor,
				   &smoothing_iterations))
    return NULL;

  smooth_vertices(varray, tarray, smoothing_factor, smoothing_iterations);
  return python_none();
}

// ----------------------------------------------------------------------------
//
static void smooth_vertices(FArray &varray, const IArray &tarray,
			    float smoothing_factor, int smoothing_iterations)
{
  int n = varray.size(0);
  IArray counts(1, &n);
  counts.set(0);
  int *c = counts.values();

  IArray tc = tarray.contiguous_array();
  int m = tarray.size();
  int *vi = tc.values();
  for (int t = 0 ; t < m ; ++t)
    c[vi[t]] += 2;

  FArray ave_neighbors(2, varray.sizes());
  float *an = ave_neighbors.values();
  float *va = varray.values();
  int s0 = varray.stride(0), s1 = varray.stride(1);
  float fv = 1 - smoothing_factor, fa = smoothing_factor;
  for (int iter = 0 ; iter < smoothing_iterations ; ++iter)
    {
      ave_neighbors.set(0);
      for (int t = 0 ; t < m ; t += 3)
	{
	  int i0 = vi[t], i1 = vi[t+1], i2 = vi[t+2];
	  int v0 = s0*i0, v1 = s0*i1, v2 = s0*i2;
	  int n0 = 3*i0, n1 = 3*i1, n2 = 3*i2;
	  for (int a = 0 ; a < 3 ; ++a)
	    {
	      int s1a = s1*a;
	      float va0 = va[v0+s1a], va1 = va[v1+s1a], va2 = va[v2+s1a];
	      an[n0+a] += va1 + va2;
	      an[n1+a] += va0 + va2;
	      an[n2+a] += va0 + va1;
	    }
	}

      for (int k = 0 ; k < n ; ++k)
	{
	  int count = c[k];
	  if (count)
	    {
	      int s0k = s0*k, k3 = 3*k;
	      for (int a = 0 ; a < 3 ; ++a)
		{
		  int v = s0k + a*s1;
		  va[v] = fv * va[v] + fa * an[k3+a] / count;
		}
	    }
	}
    }
}
