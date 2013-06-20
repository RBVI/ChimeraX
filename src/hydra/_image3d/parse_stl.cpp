// #include <iostream>			// use std::cerr for debugging.
#include <map>				// use std::map

#include <Python.h>			// use PyObject
#include <stdlib.h>			// use strncpy()

#include "pythonarray.h"		// use python_int_array

class Point
{
public:
  Point(float x, float y, float z) : x(x), y(y), z(z) {}
  bool operator<(const Point &p) const
    { return x < p.x || (x == p.x && (y < p.y || (y == p.y && z < p.z))); }
  float x,y,z;
};

// -----------------------------------------------------------------------------
// Read in an STL file and return vertices, normals and triangles.
// 
extern "C"
PyObject *parse_stl(PyObject *s, PyObject *args, PyObject *keywds)
{
  Py_buffer stl_buf;
  const char *kwlist[] = {"stl_data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("y*"),
				   (char **)kwlist, &stl_buf))
    return NULL;

  const char *stl_data = (const char *)stl_buf.buf;
  Py_ssize_t nbytes = stl_buf.len;

  if (nbytes < 84)
    {
      PyErr_SetString(PyExc_SyntaxError,
		      "read_stl(): file must be at least 84 bytes");
      return NULL;
    }

  // First read 80 byte comment line.
  char comment[81];
  comment[80] = '\0';
  strncpy(comment, stl_data, 80);
  PyObject *py_comment = PyUnicode_FromString(comment);

  // Number of triangles.
  int ntri = *(unsigned int *)(stl_data+80);
  if (nbytes != 84 + ntri*50)
    {
      PyErr_SetString(PyExc_SyntaxError,
		      "read_stl(): wrong file size");
      return NULL;
    }

  // Next read 50 bytes per triangle containing float32 normal vector
  // followed three float32 vertices, followed by two "attribute bytes"
  // sometimes used to hold color information, but ignored by this reader.
  // Eliminate identical vertices and average their normals.
  int *ta;
  PyObject *py_ta = python_int_array(ntri, 3, &ta);

  // Make triangle array using unique vertices.
  std::map<Point,int> vnum;
  for (int t = 0 ; t < ntri ; ++t)
    {
      float *tri = (float *)(stl_data+84+50*t);
      for (int c = 0 ; c < 3 ; ++c)
	{
           int c3 = 3*c + 3;
           Point p = Point(tri[c3], tri[c3+1], tri[c3+2]);
	   std::map<Point,int>::iterator pi = vnum.find(p);
	   int v;
	   if (pi == vnum.end())
	     {
               v = vnum.size();
	       vnum[p] = v;
             }
	   else
	     v = pi->second;
	   ta[3*t+c] = v;
        }
    }

  // Make vertex coordinate array.
  int nv = vnum.size();
  float *va;
  PyObject *py_va = python_float_array(nv, 3, &va);
  for (std::map<Point,int>::iterator pi = vnum.begin() ; pi != vnum.end() ; ++pi)
    {
      const Point &p = pi->first;
      int v3 = 3*pi->second;
      va[v3] = p.x;
      va[v3+1] = p.y;
      va[v3+2] = p.z;
    }

  // Make average normals array.
  float *na;
  PyObject *py_na = python_float_array(nv, 3, &na);
  int nv3 = 3*nv;
  for (int i = 0 ; i < nv3 ; ++i)
    na[i] = 0;
  for (int t = 0 ; t < ntri ; ++t)
    {
      float *tnormal = (float *)(stl_data+84+50*t);
      for (int c = 0 ; c < 3 ; ++c)
	{
          int v = ta[3*t+c];
	  for (int a = 0 ; a < 3 ; ++a)
	    na[3*v+a] += tnormal[a];
        }
    }

  // Make normals unit length.
  for (int v = 0 ; v < nv ; ++v)
    {
      int v3 = 3*v;
      float nx = na[v3], ny = na[v3+1], nz = na[v3+2];
      float nlen = sqrt(nx*nx + ny*ny + nz*nz);
      if (nlen > 0)
	{ na[v3] /= nlen; na[v3+1] /= nlen; na[v3+2] /= nlen; }
    }

  PyObject *geom = python_tuple(py_comment, py_va, py_na, py_ta);

  return geom;
}
