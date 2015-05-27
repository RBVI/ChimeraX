// ----------------------------------------------------------------------------
// Calculate vertex normals for a surface.
//
#include <math.h>			// use sqrt()

#include "pythonarray.h"		// use c_array_to_python()
#include "rcarray.h"			// use FArray, IArray

// ----------------------------------------------------------------------------
// Calculate vertex normals by adding normals for each triangle that uses
// a vertex, and then normalizing the sum.
//
FArray calculate_vertex_normals(const FArray &vertices, const IArray &triangles)
{
  int n = vertices.size(0);
  int sizes[2] = {n, 3};
  FArray normals(2, sizes);
  float *narray = normals.values();
  int n3 = n * 3;
  for (int k = 0 ; k < n3 ; ++k)
    narray[k] = 0;

  float *varray = vertices.values();
  int *tarray = triangles.values();
  int tsize = 3 * triangles.size(0);
  for (int ei = 0 ; ei < tsize ; ei += 3)
    {
      int i1 = 3 * tarray[ei];
      int i2 = 3 * tarray[ei+1];
      int i3 = 3 * tarray[ei+2];
      float v1x = varray[i1], v1y = varray[i1+1], v1z = varray[i1+2];
      float v2x = varray[i2], v2y = varray[i2+1], v2z = varray[i2+2];
      float v3x = varray[i3], v3y = varray[i3+1], v3z = varray[i3+2];
      float e1x = v2x - v1x, e1y = v2y - v1y, e1z = v2z - v1z;
      float e2x = v3x - v1x, e2y = v3y - v1y, e2z = v3z - v1z;
      float nx = (e1y*e2z - e1z*e2y);
      float ny = (-e1x*e2z + e1z*e2x);
      float nz = (e1x*e2y - e1y*e2x);
      float norm = sqrt(nx*nx + ny*ny + nz*nz);
      if (norm > 0)
	{
	  nx /= norm; ny /= norm; nz /= norm;
	  narray[i1] += nx; narray[i1+1] += ny; narray[i1+2] += nz;
	  narray[i2] += nx; narray[i2+1] += ny; narray[i2+2] += nz;
	  narray[i3] += nx; narray[i3+1] += ny; narray[i3+2] += nz;
	}
    }

  for (int k = 0 ; k < n3 ; k += 3)
    {
      float nx = narray[k], ny = narray[k+1], nz = narray[k+2];
      float norm = sqrt(nx*nx + ny*ny + nz*nz);
      if (norm == 0)
	narray[k] = 1;
      else
	{
	  narray[k] /= norm; narray[k+1] /= norm; narray[k+2] /= norm;
	}
    }

  return normals;
}

// ----------------------------------------------------------------------------
// Flip normals and reverse triangle vertex order.
//
void invert_vertex_normals(const FArray &normals, const IArray &triangles)
{
  int n = normals.size(0), ns0 = normals.stride(0), ns1 = normals.stride(1);
  float *narray = normals.values();
  for (int k = 0 ; k < n ; ++k)
    {
      int i = ns0*k;
      for (int a = 0 ; a < 3 ; ++a, i+= ns1)
	narray[i] = -narray[i];
    }

  int t = triangles.size(0);
  int ts0 = triangles.stride(0), ts2 = 2*triangles.stride(1);
  int *tarray = triangles.values();
  for (int k = 0 ; k < t ; ++k)
    {
      int i = ts0*k, i2 = i + ts2;
      int v0 = tarray[i];
      tarray[i] = tarray[i2];
      tarray[i2] = v0;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *calculate_vertex_normals(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray va;
  IArray ta;
  const char *kwlist[] = {"vertices", "triangles", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_float_n3_array, &va,
				   parse_int_n3_array, &ta))
    return NULL;

  FArray na = calculate_vertex_normals(va, ta);
  PyObject *normals = c_array_to_python(na.values(), na.size(0), na.size(1));
  return normals;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *invert_vertex_normals(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray na;
  IArray ta;
  const char *kwlist[] = {"normals", "triangles", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&"), (char **)kwlist,
				   parse_writable_float_n3_array, &na,
				   parse_writable_int_n3_array, &ta))
    return NULL;

  invert_vertex_normals(na, ta);

  Py_INCREF(Py_None);
  return Py_None;
}
