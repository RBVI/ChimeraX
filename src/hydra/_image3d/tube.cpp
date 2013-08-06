// ----------------------------------------------------------------------------
//
#include <Python.h>			// use PyObject

#include "pythonarray.h"		// use array_from_python()
#include "rcarray.h"			// use FArray, IArray

static void vector_rotation(float *v1, float *v2, float *r);
static void transform(float *points, int m, float *rot, float *offset, float *result);
static void matrix_multiply(float *a, float *b, float *c);
static void stitch_sections(int n, int m, int *triangles);
static void stitch_cap(int *triangles, int m, int voffset, bool clockwise);

// -----------------------------------------------------------------------------
// Create triangle geometry for a tube with given center-line path and cross-section.
//
// TODO: Add tube end-caps.
//
static void tube(float *path, float *tangents, int n,
		 float *cross_section, float *cross_section_normals, int m,
		 bool end_caps,
		 float *vertices, float *normals, int *triangles)
{
  float t[3] = {0,0,1}, zero_offset[3] = {0,0,0};
  float r[9], cr[9] = {1,0,0, 0,1,0, 0,0,1};
  for (int i = 0 ; i < n ; ++i)
    {
      int i3 = 3*i;
      vector_rotation(t, tangents + i3, r);
      matrix_multiply(r, cr, cr);
      t[0] = tangents[i3]; t[1] = tangents[i3+1]; t[2] = tangents[i3+2];
      transform(cross_section, m, cr, path + i3, vertices + i3*m);
      transform(cross_section_normals, m, cr, zero_offset, normals + i3*m);
    }
  stitch_sections(n, m, triangles);

  if (end_caps)
    {
      int m3 = 3*m, e = 3*(n-1)*m, c1 = 3*n*m, c2 = 3*(n+1)*m;
      float *t2 = tangents + 3*(n-1);
      for (int i = 0 ; i < m3 ; ++i)
	{
	  vertices[c1+i] = vertices[i];
	  normals[c1+i] = -tangents[i%3];
	  vertices[c2+i] = vertices[e+i];
	  normals[c2+i] = t2[i%3];
	}
      stitch_cap(triangles+3*2*(n-1)*m, m, n*m, false);
      stitch_cap(triangles+3*2*(n-1)*m + 3*(m-2), m, (n+1)*m, true);
    }
}


// -----------------------------------------------------------------------------
// Left multiply rotation matrix taking unit vector n0 to unit vector n1.
//
static void vector_rotation(float *v1, float *v2, float *r)
{
  float c = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
  if (c <= -1)
    {
      r[0] = r[4] = -1; r[8] = 1;
      r[1] = r[2] = r[3] = r[5] = r[6] = r[7] = 0;
      return;
    }

  float wx = v1[1]*v2[2]-v1[2]*v2[1], wy = v1[2]*v2[0]-v1[0]*v2[2], wz = v1[0]*v2[1]-v1[1]*v2[0];
  float c1 = 1.0/(1+c);
  float cx = c1*wx, cy = c1*wy, cz = c1*wz;
  r[0] = cx*wx + c; r[1] = cx*wy - wz; r[2] = cx*wz + wy;
  r[3] = cy*wx + wz; r[4] = cy*wy + c; r[5] = cy*wz - wx;
  r[6] = cz*wx - wy; r[7] = cz*wy + wx; r[8] = cz*wz + c;
}

// -----------------------------------------------------------------------------
// Apply rotation and translation to points.
//
static void transform(float *points, int m, float *rot, float *offset, float *result)
{
  int m3 = 3*m;
  for (int i = 0 ; i < m3 ; i += 3)
    {
      float x = points[i], y = points[i+1], z = points[i+2];
      result[i] = rot[0]*x + rot[1]*y + rot[2]*z + offset[0];
      result[i+1] = rot[3]*x + rot[4]*y + rot[5]*z + offset[1];
      result[i+2] = rot[6]*x + rot[7]*y + rot[8]*z + offset[2];
    }
}

// -----------------------------------------------------------------------------
// Multiply 3x3 matrices.  Result c may be same array as a or b.
//
static void matrix_multiply(float *a, float *b, float *c)
{
  float c00 = a[0]*b[0] + a[1]*b[3] + a[2]*b[6];
  float c01 = a[0]*b[1] + a[1]*b[4] + a[2]*b[7];
  float c02 = a[0]*b[2] + a[1]*b[5] + a[2]*b[8];
  float c10 = a[3]*b[0] + a[4]*b[3] + a[5]*b[6];
  float c11 = a[3]*b[1] + a[4]*b[4] + a[5]*b[7];
  float c12 = a[3]*b[2] + a[4]*b[5] + a[5]*b[8];
  float c20 = a[6]*b[0] + a[7]*b[3] + a[8]*b[6];
  float c21 = a[6]*b[1] + a[7]*b[4] + a[8]*b[7];
  float c22 = a[6]*b[2] + a[7]*b[5] + a[8]*b[8];
  c[0] = c00; c[1] = c01; c[2] = c02;
  c[3] = c10; c[4] = c11; c[5] = c12;
  c[6] = c20; c[7] = c21; c[8] = c22;
}

// -----------------------------------------------------------------------------
// Make triangulation for n sections each having m points
//
static void stitch_sections(int n, int m, int *triangles)
{
  int t = 0;
  for (int i = 0 ; i < n-1 ; ++i)
    {
      int mi = m*i, mi1 = mi + m;
      for (int j = 0 ; j < m ; ++j)
	{
	  int j1 = (j+1)%m;
	  triangles[t] = mi+j;
	  triangles[t+1] = mi + j1;
	  triangles[t+2] = mi1 + j;
	  triangles[t+3] = mi1 + j;
	  triangles[t+4] = mi + j1;
	  triangles[t+5] = mi1 + j1;
	  t += 6;
	}
    }
}

// -----------------------------------------------------------------------------
// Make triangle strip for closed path of m vertices.
//
static void stitch_cap(int *triangles, int m, int voffset, bool clockwise)
{
  int t = 0, o = voffset;
  int t1 = (clockwise ? 1 : 2), t2 = (clockwise ? 2 : 1);
  for (int i = 1, j = m-2 ; i <= j ; ++i, --j)
    {
      triangles[t] = o+j+1;
      triangles[t+t1] = o+i-1;
      triangles[t+t2] = o+i;
      if (i == j)
	break;
      t += 3;
      triangles[t] = o+j+1;
      triangles[t+t1] = o+i;
      triangles[t+t2] = o+j;
      t += 3;
    }
}

// -----------------------------------------------------------------------------
//
extern "C"
PyObject *tube_geometry(PyObject *s, PyObject *args, PyObject *keywds)
{
  FArray path, tangents, cross_section, cross_section_normals;
  const char *kwlist[] = {"path", "tangents", "cross_section", "cross_section_normals", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &path,
				   parse_float_n3_array, &tangents,
				   parse_float_n3_array, &cross_section,
				   parse_float_n3_array, &cross_section_normals))
    return NULL;

  int n = path.size(0);
  if (tangents.size(0) != n)
    {
      PyErr_SetString(PyExc_ValueError,
		      "tube_geometry(): path and tangent arrays have differing size");
      return NULL;
    }
  float *p = path.values(), *t = tangents.values();

  int m = cross_section.size(0);
  if (cross_section_normals.size(0) != m)
    {
      PyErr_SetString(PyExc_ValueError,
		      "tube_geometry(): cross section and cross section normals "
		      "arrays have differing size");
      return NULL;
    }
  float *cs = cross_section.values(), *csn = cross_section_normals.values();

  float *vertices, *normals;
  int *triangles;
  PyObject *vertices_py = python_float_array((n+2)*m, 3, &vertices);
  PyObject *normals_py = python_float_array((n+2)*m, 3, &normals);
  PyObject *triangles_py = python_int_array(2*(n-1)*m+2*(m-2), 3, &triangles);

  tube(p, t, n, cs, csn, m, true, vertices, normals, triangles);

  PyObject *pt = python_tuple(vertices_py, normals_py, triangles_py);
  return pt;
}
