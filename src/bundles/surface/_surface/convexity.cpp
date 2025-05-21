// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
//
#include <iostream>		// use std::cerr for debugging
#include <map>			// use std::map
#include <utility>		// use std::pair
#include <vector>		// use std::vector

#include <math.h>		// use sqrt, acos
#include <Python.h>		// use PyObject

#include <arrays/pythonarray.h>	// use parse_*()
#include <arrays/rcarray.h>	// use IArray, FArray, DArray

typedef std::pair<int, int> Edge;
typedef std::vector<int> TriangleIndices;
typedef std::map<Edge, TriangleIndices> EdgeTriangles;
typedef std::vector<int> VertexIndices;
typedef std::vector<VertexIndices> VertexNeighbors;

static void convexity(const FArray &varray, const IArray &tarray, int smoothing_iterations, DArray &cvalues);
static double bend_angle(float *n1, float *n2, float *e);
static FArray triangle_normals(const FArray &varray, const IArray &tarray);
static void edge_triangles(const IArray &tarray, EdgeTriangles &et);
static void smooth_surface_values(const FArray &varray, EdgeTriangles &edges,
				  DArray &values, int smoothing_iterationsy);
static int *unique_vertices(FArray varray);
static IArray nondegenerate_triangles(IArray tarray, int *vmap);

// ----------------------------------------------------------------------------
//
extern "C" PyObject *vertex_convexity(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray varray;
  IArray tarray;
  int smoothing_iterations = 0;
  DArray carray;
  const char *kwlist[] = {"vertices", "triangles", "smoothing_iterations", "convexity", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&|iO&"), (char **)kwlist,
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   &smoothing_iterations,
				   parse_writable_double_n_array, &carray))
    return NULL;

  PyObject *result;
  if (carray.dimension() == 0)
    {
      double *c;
      int nv = varray.size(0);
      result = python_double_array(nv, &c);
      for (int i = 0 ; i < nv ; ++i)
	c[i] = 0;
      parse_writable_double_n_array(result, &carray);
    }
  else
    result = python_none();

  //convexity(varray, tarray, smoothing_iterations, carray);

  int *vmap = unique_vertices(varray);
  if (vmap == NULL)
    convexity(varray, tarray, smoothing_iterations, carray);
  else
    {
      // Convexity calculation requires triangles with normals.
      // Eliminate zero-area triangles.
      IArray ndt = nondegenerate_triangles(tarray, vmap);
      convexity(varray, ndt, smoothing_iterations, carray);
      size_t n = varray.size(0);
      double *ca = carray.values();
      int64_t cs0 = carray.stride(0);
      for (size_t i = 0 ; i < n ; ++i)
	ca[i*cs0] = ca[vmap[i]*cs0];
      delete [] vmap;
    }

  return result;
}

// ----------------------------------------------------------------------------
//
static void convexity(const FArray &varray, const IArray &tarray, int smoothing_iterations,
		      DArray &cvalues)
{
  FArray tnormals = triangle_normals(varray, tarray);
  float *tn = tnormals.values();
  EdgeTriangles et;
  edge_triangles(tarray, et);

  float *va = varray.values();
  int64_t vs0 = varray.stride(0), vs1 = varray.stride(1);
  double *ca = cvalues.values();
  int64_t cs0 = cvalues.stride(0);
  float e[3];
  // Cone angle is sphere area which equals n*pi - sum of bend angles of spherical polygon.
  for (auto eti = et.begin() ; eti != et.end() ; ++eti)
    {
      TriangleIndices &tri = eti->second;
      if (tri.size() != 2)
	continue;
      int t1 = tri[0], t2 = tri[1];
      const Edge &ed = eti->first;
      int v1 = ed.first, v2 = ed.second;
      for (int i = 0 ; i < 3 ; ++i)
	e[i] = va[vs0*v2+vs1*i]-va[vs0*v1+vs1*i];
      double a = bend_angle(tn+3*t1, tn+3*t2, e);
      ca[cs0*v1] += a;
      ca[cs0*v2] += a;
    }

  if (smoothing_iterations > 0)
    smooth_surface_values(varray, et, cvalues, smoothing_iterations);
}

// ----------------------------------------------------------------------------
//
static double bend_angle(float *n1, float *n2, float *e)
{
  double i12 = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
  if (i12 < -1)
    i12 = -1;
  else if (i12 > 1)
    i12 = 1;
  double a = acos(i12);
  float c12[3] = {n1[1]*n2[2]-n1[2]*n2[1], n1[2]*n2[0]-n1[0]*n2[2], n1[0]*n2[1]-n1[1]*n2[0]};
  if (c12[0]*e[0] + c12[1]*e[1] + c12[2]*e[2] < 0)
    a = -a;
  return a;
}

// ----------------------------------------------------------------------------
//
static FArray triangle_normals(const FArray &varray, const IArray &tarray)
{
  float *va = varray.values();
  int64_t vs0 = varray.stride(0), vs1 = varray.stride(1);
  int *ta = tarray.values();
  int64_t ts0 = tarray.stride(0), ts1 = tarray.stride(1);

  int64_t nt = tarray.size(0);
  int64_t size[2] = {nt, 3};
  FArray tn(2, size);
  float *tna = tn.values();
  for (int t = 0 ; t < nt ; ++t)
    {
      int v0 = ta[ts0*t], v1 = ta[ts0*t+ts1], v2 = ta[ts0*t+2*ts1];
      float *v0a = va + vs0*v0, *v1a = va + vs0*v1, *v2a = va + vs0*v2;
      float x0 = v0a[0], y0 = v0a[vs1], z0 = v0a[2*vs1];
      float x1 = v1a[0], y1 = v1a[vs1], z1 = v1a[2*vs1];
      float x2 = v2a[0], y2 = v2a[vs1], z2 = v2a[2*vs1];
      float v01x = x1-x0, v01y = y1-y0, v01z = z1-z0;
      float v02x = x2-x0, v02y = y2-y0, v02z = z2-z0;
      float c12x = v01y*v02z-v01z*v02y, c12y = v01z*v02x-v01x*v02z, c12z = v01x*v02y-v01y*v02x;
      float n = sqrt(c12x*c12x + c12y*c12y + c12z*c12z);
      if (n == 0) {
	n = 1;
	std::cerr << "Zero area triangle will produce wrong convexity values." << std::endl;
      }
      tna[3*t] = c12x/n;
      tna[3*t+1] = c12y/n;
      tna[3*t+2] = c12z/n;
    }
  return tn;
}

// ----------------------------------------------------------------------------
//
static void edge_triangles(const IArray &tarray, EdgeTriangles &et)
{
  int *ta = tarray.values();
  int64_t ts0 = tarray.stride(0), ts1 = tarray.stride(1);
  int nt = tarray.size(0);
  int v[3];
  for (int t = 0 ; t < nt ; ++t)
    {
      v[0] = ta[ts0*t]; v[1] = ta[ts0*t+ts1]; v[2] = ta[ts0*t+2*ts1];
      for (int e = 0 ; e < 3 ; ++e)
	{
	  int va = v[e], vb = v[(e+1)%3];
	  Edge eba(vb, va);
	  EdgeTriangles::iterator ei = et.find(eba);
	  if (ei != et.end())
	    ei->second.push_back(t);
	  else
	    {
	      Edge eab(va,vb);
	      et[eab].push_back(t);
	    }
	}
    }
}
 
// ----------------------------------------------------------------------------
//
static void smooth_surface_values(const FArray &varray, EdgeTriangles &edges,
				  DArray &values, int smoothing_iterations)
{
  int nv = varray.size(0);
  VertexNeighbors vn(nv);
  for (int i = 0 ; i < nv ; ++i)
    vn[i].push_back(i);
  for (auto eti = edges.begin() ; eti != edges.end() ; ++eti)
    {
      const Edge &e = eti->first;
      int va = e.first, vb = e.second;
      vn[va].push_back(vb);
      vn[vb].push_back(va);
    }

  double *vals = values.values();
  int64_t vs0 = values.stride(0);
  double *values2 = new double [nv];
  for (int r = 0 ; r < smoothing_iterations ; ++r)
    {
      for (int i = 0 ; i < nv ; ++i)
	{
	  VertexIndices &vi = vn[i];
	  size_t nvi = vi.size();
	  double vsum = 0;
	  for (size_t j = 0 ; j < nvi ; ++j)
	    vsum += vals[vs0*vi[j]];
	  values2[i] = vsum / nvi;
	}
      for (int i = 0 ; i < nv ; ++i)
        vals[vs0*i] = values2[i];
    }
}
 
// ----------------------------------------------------------------------------
//
class Vertex
{
public:
  Vertex() {}
  bool operator<(const Vertex &v) const
    { return x < v.x || (x == v.x && (y < v.y || (y == v.y && z < v.z))); }
  float x,y,z;
};
typedef std::map<Vertex, int> VertexIndex;

// ----------------------------------------------------------------------------
//
static int *unique_vertices(FArray varray)
{
  VertexIndex vi;
  size_t nv = varray.size(0);
  int *vmap = new int [nv];
  float *va = varray.values();
  int64_t vs0 = varray.stride(0), vs1 = varray.stride(1);
  Vertex v;
  for (size_t i = 0 ; i < nv ; ++i, va += vs0)
    {
      v.x = va[0]; v.y = va[vs1]; v.z = va[2*vs1];
      VertexIndex::iterator j = vi.find(v);
      if (j == vi.end())
	vi[v] = vmap[i] = i;
      else
	vmap[i] = j->second;
    }
  if (vi.size() == nv)
    {
      delete [] vmap;
      vmap = NULL;
    }
  return vmap;
}

// ----------------------------------------------------------------------------
//
static IArray nondegenerate_triangles(IArray tarray, int *vmap)
{
  int *ta = tarray.values();
  int64_t ts0 = tarray.stride(0), ts1 = tarray.stride(1);
  int64_t nt = tarray.size(0);
  int nnd = 0;
  for (int64_t t = 0 ; t < nt ; ++t, ta += ts0)
    {
      int v0 = vmap[ta[0]], v1 = vmap[ta[ts1]], v2 = vmap[ta[2*ts1]];
      if (v0 != v1 && v0 != v2 && v1 != v2)
	nnd += 1;
    }
  int64_t size[2] = {nnd, 3};
  IArray tnd(2, size);
  int *tnda = tnd.values();
  ta = tarray.values();
  for (int64_t t = 0 ; t < nt ; ++t, ta += ts0)
    {
      int v0 = vmap[ta[0]], v1 = vmap[ta[ts1]], v2 = vmap[ta[2*ts1]];
      if (v0 != v1 && v0 != v2 && v1 != v2)
	{ *tnda++ = v0; *tnda++ = v1; *tnda++ = v2; }
    }
  return tnd;
}
