// ----------------------------------------------------------------------------
// Triangulate a set of loops in a plane.
//
// Uses gluTess*() routines from OpenGL.
//
// In the OpenGL 1.3 implementations tested this uses a sweepline
// technique that generates long skinny triangles.
//
#ifdef __APPLE__
#include <OpenGL/glu.h>			// use gluTess*()
#else
#include <GL/glu.h>			// use gluTess*()
#endif

#include <Python.h>			// use PyObject

#include "border.h"			// Use Vertices, Triangles, Loops
#include "pythonarray.h"		// use parse_float_n3_array(), ...
#include "rcarray.h"			// use Array<float>

namespace Cap_Calculation
{

// ----------------------------------------------------------------------------
//
class Polygon_Data
{
public:
  Polygon_Data(Vertices &p, Triangles &t)
    : points(p), triangles(t)
    {}
  Vertices &points;
  Triangles &triangles;
};

// ----------------------------------------------------------------------------
//
extern "C"
{
typedef void (*GLU_func)();
}

// ----------------------------------------------------------------------------
//
static void triangulate_polygon(Loops &loops, float normal[3],
				Polygon_Data &polygon_data);
extern "C"
{
static void edge_flag_callback(GLboolean);
static void vertex_data_callback(void *vertex_data,
				    void *polygon_data);
static void combine_data_callback(GLdouble coords[3],
				    void *vertex_data[4],
				    GLfloat weight[4], void **outData,
				    void *polygon_data);
}

// ----------------------------------------------------------------------------
// Triangulate polygonal region bounded by loops.
//
void triangulate_polygon(Loops &loops, float normal[3],
			 Vertices &vertex_positions,
			 Triangles &triangle_vertex_indices)
{
  Polygon_Data polygon_data(vertex_positions, triangle_vertex_indices);
  triangulate_polygon(loops, normal, polygon_data);
}

// ----------------------------------------------------------------------------
// Triangulate polygonal region bounded by loops.
//
static void triangulate_polygon(Loops &loops, float normal[3],
				Polygon_Data &polygon_data)
{
  // Use GLU tesselator routines to triangulate polygon.
  GLUtesselator *tess = gluNewTess();
  // The edge flag callback forces output to be independent triangles instead
  // of strips and fans.
  gluTessCallback(tess, GLU_TESS_EDGE_FLAG,
		  reinterpret_cast<GLU_func>(&edge_flag_callback));
  gluTessCallback(tess, GLU_TESS_VERTEX_DATA,
		  reinterpret_cast<GLU_func>(&vertex_data_callback));
  gluTessCallback(tess, GLU_TESS_COMBINE_DATA,
		  reinterpret_cast<GLU_func>(&combine_data_callback));
  gluTessNormal(tess, normal[0], normal[1], normal[2]);
  gluTessBeginPolygon(tess, &polygon_data);
  for (unsigned int l = 0 ; l < loops.size() ; ++l)
    {
      int kbegin = loops[l].first, klast = loops[l].second;
      gluTessBeginContour(tess);
      // Loop over vertices
      for (int k = kbegin ; k <= klast ; ++k)
	{
	  int k3 = 3*k;
	  // Tesselator needs double point location;
	  GLdouble point[3];
	  for (int a = 0 ; a < 3 ; ++a)
	    point[a] = polygon_data.points[k3+a];
	  gluTessVertex(tess, point, reinterpret_cast<void *>(k));
	}
      gluTessEndContour(tess);
    }
  gluTessEndPolygon(tess);
  gluDeleteTess(tess);
}

extern "C"
{

// ----------------------------------------------------------------------------
//
static void edge_flag_callback(GLboolean)
{
  // The edge flag callback forces output to be independent triangles instead
  // of strips and fans.  We ignore the edge flag.
}

// ----------------------------------------------------------------------------
// Save tesselation triangle vertex.
//
static void vertex_data_callback(void *vertex_data, void *polygon_data)
{
  Polygon_Data *p = static_cast<Polygon_Data *>(polygon_data);
  int k = static_cast<int>(reinterpret_cast<long>(vertex_data));
  p->triangles.push_back(k);
}

// ----------------------------------------------------------------------------
// Add a new vertex needed for tesselation.
//
static void combine_data_callback(GLdouble coords[3],
				    void *vertex_data[4],
				    GLfloat weight[4], void **outData,
				    void *polygon_data)
{
  Polygon_Data *p = static_cast<Polygon_Data *>(polygon_data);
  float x = static_cast<float>(coords[0]);
  float y = static_cast<float>(coords[1]);
  float z = static_cast<float>(coords[2]);
  p->points.push_back(x);
  p->points.push_back(y);
  p->points.push_back(z);
  int k = (p->points.size()/3) - 1;
  *outData = reinterpret_cast<void *>(k);
}

}	// end of extern "C"

}	// end of namespace Cap_Calculation

// ----------------------------------------------------------------------------
//
static bool convert_loops(const IArray &loops, int nv,
			  Cap_Calculation::Loops *ccloops)
{
  int n = loops.size(0), s0 = loops.stride(0), s1 = loops.stride(1);
  int *la = loops.values();
  for (int k = 0 ; k < n ; ++k)
    {
      int s = la[k*s0], e = la[k*s0+s1];
      if (s < 0 || s >= nv || e < 0 || e >= nv)
	{
	  PyErr_SetString(PyExc_ValueError, "Loop vertex range out of bounds");
	  return false;
	}
      ccloops->push_back(Cap_Calculation::Loop(s,e));
    }
  return true;
}

// ----------------------------------------------------------------------------
// Returns no triangles on failure.
//
extern "C" PyObject *triangulate_polygon(PyObject *, PyObject *args, PyObject *keywds)
{
  IArray loops;
  float normal[3];
  FArray varray;
  const char *kwlist[] = {"loops", "normal", "vertices", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&"), (char **)kwlist,
				   parse_int_n2_array, &loops,
				   parse_float_3_array, normal,
				   parse_float_n3_array, &varray))
    return NULL;

  Cap_Calculation::Loops ccloops;
  if (!convert_loops(loops, varray.size(0), &ccloops))
    return NULL;

  std::vector<float> v;
  int n = varray.size(0), s0 = varray.stride(0), s1 = varray.stride(1);
  float *va = varray.values();
  for (int k = 0 ; k < n ; ++k)
    {
      v.push_back(va[k*s0]);
      v.push_back(va[k*s0+s1]);
      v.push_back(va[k*s0+2*s1]);
    }

  std::vector<int> t;
  Cap_Calculation::triangulate_polygon(ccloops, normal, v, t);

  if ((int)v.size() > 3*n)
    t.clear();		// Vertex added.

  int *ti = (t.size() == 0 ? NULL : &t.front());
  PyObject *tarray_py = c_array_to_python(ti, t.size()/3, 3);

  return tarray_py;
}
