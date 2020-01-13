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
// Calculate surfaces surrounding voxels with certain values of a volume.
//
#include <iostream>		// use std::cerr for debugging

#include <map>
#include <vector>

#include <arrays/pythonarray.h>		// use c_array_to_python()
#include <arrays/rcarray.h>		// use call_template_function()

using Reference_Counted_Array::Array;

namespace Segment_Surface
{
  
// ----------------------------------------------------------------------------
//
class Vertices
{
public:
  int count() { return vertices.size()/3; }
  int add(float x, float y, float z)
  {
    int i = vertices.size() / 3;
    vertices.push_back(x);
    vertices.push_back(y);
    vertices.push_back(z);
    return i;
  }
  float *values() { return &vertices[0]; }
private:
  std::vector<float> vertices;
};
  
// ----------------------------------------------------------------------------
//
class Triangles
{
public:
  int count() { return triangles.size()/3; }
  int add(int v0, int v1, int v2)
  {
    int i = triangles.size() / 3;
    triangles.push_back(v0);
    triangles.push_back(v1);
    triangles.push_back(v2);
    return i;
  }
  int *values() { return &triangles[0]; }
private:
  std::vector<int> triangles;
};

class VertexMap;
  
// ----------------------------------------------------------------------------
//
class Surface
{
public:
  Surface(int id = 0)
  {
    this->_id = id;
  }
  int add_vertex(float x, float y, float z)
  {
    return vertices.add(x, y, z);
  }
  int vertex_count()
  {
    return vertices.count();
  }
  void add_square(int v0, int v1, int v2, int v3, int side)
  {
    if (side)
      {
	triangles.add(v0,v1,v2);
	triangles.add(v0,v2,v3);
      }
    else
      {
	triangles.add(v0,v2,v1);
	triangles.add(v0,v3,v2);
      }
  }
  inline void add_voxel_face(int ix, int iy, int iz, int axis, int side, VertexMap &vmap);
  int id() const
  {
    return _id;
  }
  PyObject *python_vertex_array()
  {
    return c_array_to_python(vertices.values(), vertices.count(), 3);
  }
  PyObject *python_triangle_array()
  {
    return c_array_to_python(triangles.values(), triangles.count(), 3);
  }
  
private:
  int _id;
  Vertices vertices;
  Triangles triangles;
};

typedef std::vector<Surface *> Surfaces;
  
// ----------------------------------------------------------------------------
//
class VertexMap
{
public:
  VertexMap(int width, int height)
  {
    this->width = width;
    this->height = height;
    int psize = width * height;
    this->vplane0 = new int[psize];
    this->vplane1 = new int[psize];
    reset();
  }
  ~VertexMap()
  {
    delete [] vplane1;
    delete [] vplane0;
  }

  void reset()
  {
    int psize = width * height;
    for (int i = 0 ; i < psize ; ++i)
      vplane0[i] = vplane1[i] = -1;
    this->vbase0 = 0;
    this->vbase1 = 0;
    this->iz0 = 0;
  }
  
  int vertex(int ix, int iy, int iz, Surface &surface)
  {
      int *vp, vbase;
      if (iz == iz0)
	{
	  vp = vplane0;
	  vbase = vbase0;
	}
      else
	{
	  if (iz > iz0 + 1)
	    next_plane(iz, surface.vertex_count());
	  vp = vplane1;
	  vbase = vbase1;
	}
      
      int i = ix + iy*width;
      vp += i;
      int v = *vp;

      if (v < vbase)
	*vp = v = surface.add_vertex(ix-0.5,iy-0.5,iz-0.5);
      return v;
  }

  void next_plane(int iz, int vbase)
  {
    if (iz == iz0+2)
      {
	// Set the new plane 0 to be the current plane 1.
	int *temp = vplane0;
	vplane0 = vplane1;
	vbase0 = vbase1;
	vplane1 = temp;
	vbase1 = vbase;
	iz0 += 1;
      }
    else if (iz > iz0+2)
      {
	vbase0 = vbase1 = vbase;
	iz0 = iz-1;
      }
  }
  
private:
  int *vplane0, *vplane1;
  int width, height;
  int vbase0, vbase1;
  int iz0;
};

void Surface::add_voxel_face(int ix, int iy, int iz, int axis, int side, VertexMap &vmap)
{
  int v0, v1, v2, v3;
  if (axis == 0)
    {
	v0 = vmap.vertex(ix+side, iy, iz, *this);
	v1 = vmap.vertex(ix+side, iy+1, iz, *this);
	v2 = vmap.vertex(ix+side, iy+1, iz+1, *this);
	v3 = vmap.vertex(ix+side, iy, iz+1, *this);
    }
  else if (axis == 1)
    {
	v0 = vmap.vertex(ix, iy+side, iz, *this);
	v1 = vmap.vertex(ix, iy+side, iz+1, *this);
	v2 = vmap.vertex(ix+1, iy+side, iz+1, *this);
	v3 = vmap.vertex(ix+1, iy+side, iz, *this);
    }
  else if (axis == 2)
    {
	v0 = vmap.vertex(ix, iy, iz+side, *this);
	v1 = vmap.vertex(ix+1, iy, iz+side, *this);
	v2 = vmap.vertex(ix+1, iy+1, iz+side, *this);
	v3 = vmap.vertex(ix, iy+1, iz+side, *this);
    }
  else
    throw std::runtime_error("add_voxel_face() error: axis is not 0,1, or 2");

  add_square(v0, v1, v2, v3, side);
}
  
// ----------------------------------------------------------------------------
//
template <class T>
void segment_surface(const Array<T> &image, T value, Surface &surface)
{
  // axes x,y,z = 2,1,0
  int xsize = image.size(2), ysize = image.size(1), zsize = image.size(0);
  long xstride = image.stride(2), ystride = image.stride(1), zstride = image.stride(0);

  // std::cerr << "size " << size0 << " " << size1 << " " << size2 << std::endl;
  // std::cerr << "stride " << xstride << " " << ystride << " " << zstride << std::endl;
  // Allocate two plane arrays to record vertex indices.
  VertexMap vmap(xsize+1, ysize+1);

  // Create squares along each of 3 axes at each segment boundary point.
  T *im = image.values();
  int xsm = xsize - 1, ysm = ysize - 1, zsm = zsize - 1;
  for (int iz = 0 ; iz < zsize ; ++iz, im += zstride - ysize*ystride)
    for (int iy = 0 ; iy < ysize ; ++iy, im += ystride - xsize*xstride)
      for (int ix = 0 ; ix < xsize ; ++ix, im += xstride)
	  {
	    if (*im != value)
	      continue;

	    if (ix == 0 || *(im - xstride) != value)  // add -x face
	      surface.add_voxel_face(ix,iy,iz, 0, 0, vmap);

	    if (ix == xsm || *(im + xstride) != value)  // add +x face
	      surface.add_voxel_face(ix,iy,iz, 0, 1, vmap);

	    if (iy == 0 || *(im - ystride) != value)  // add -y face
	      surface.add_voxel_face(ix,iy,iz, 1, 0, vmap);

	    if (iy == ysm || *(im + ystride) != value)  // add +y face
	      surface.add_voxel_face(ix,iy,iz, 1, 1, vmap);

	    if (iz == 0 || *(im - zstride) != value)  // add -z face
	      surface.add_voxel_face(ix,iy,iz, 2, 0, vmap);

	    if (iz == zsm || *(im + zstride) != value)  // add +z face
	      surface.add_voxel_face(ix,iy,iz, 2, 1, vmap);
	  }
}

// ----------------------------------------------------------------------------
//
template <class T>
int voxel_id(T v, int *surf_ids, int nids, long id_stride)
{
  return ((v < 0 || (int)v >= nids) ? 0 : surf_ids[(int)v * id_stride]);
}

// ----------------------------------------------------------------------------
//
class Face
{
public:
  Face(int ix, int iy, int iz, int axis, int side)
    : ix(ix), iy(iy), iz(iz), axis(axis), side(side) {}
  // TODO: use smaller data types to save memory.
  int ix, iy, iz, axis, side;
};

// ----------------------------------------------------------------------------
//
class FaceList
{
public:
  void add_face(int ix, int iy, int iz, int axis, int side)
  {
    faces.push_back(Face(ix, iy, iz, axis, side));
  }
  Surface *make_surface(int id, VertexMap &vmap)
  {
    Surface *surf = new Surface(id);
    for (size_t i = 0 ; i < faces.size() ; ++i)
      {
	const Face &f = faces[i];
	surf->add_voxel_face(f.ix, f.iy, f.iz, f.axis, f.side, vmap);
      }
    return surf;
  }
private:
  std::vector<Face> faces;
};
typedef std::map<int, FaceList *> FaceListMap;

// ----------------------------------------------------------------------------
//
class FaceLists
{
public:
  ~FaceLists()
  {
    for (FaceListMap::iterator fi = id_faces.begin() ; fi != id_faces.end() ; ++fi)
      delete fi->second;
    id_faces.clear();
  }
  void add_face(int id, int ix, int iy, int iz, int axis, int side)
  {
    FaceListMap::iterator fi = id_faces.find(id);
    FaceList *fl;
    if (fi == id_faces.end())
      {
	fl = new FaceList();
	id_faces[id] = fl;
      }
    else
      fl = fi->second;
    fl->add_face(ix, iy, iz, axis, side);
  }
  void make_surfaces(Surfaces &surfaces, VertexMap &vmap)
  {
    for (FaceListMap::iterator fi = id_faces.begin() ; fi != id_faces.end() ; ++fi)
      {
	surfaces.push_back(fi->second->make_surface(fi->first, vmap));
	vmap.reset();
      }
  }
  
private:
  FaceListMap id_faces;
};

// ----------------------------------------------------------------------------
//
template <class T>
void segment_surfaces(const Array<T> &image, bool surface_zero, Surfaces &surfaces)
{
  // axes x,y,z = 2,1,0
  int xsize = image.size(2), ysize = image.size(1), zsize = image.size(0);
  long xstride = image.stride(2), ystride = image.stride(1), zstride = image.stride(0);

  // Create squares along each of 3 axes at each segment boundary point.
  FaceLists faces;
  T *im = image.values();
  int xsm = xsize - 1, ysm = ysize - 1, zsm = zsize - 1;
  for (int iz = 0 ; iz < zsize ; ++iz, im += zstride - ysize*ystride)
    for (int iy = 0 ; iy < ysize ; ++iy, im += ystride - xsize*xstride)
      for (int ix = 0 ; ix < xsize ; ++ix, im += xstride)
	{
	    T v = *im;
	    if (!surface_zero && v == 0)
	      continue;
	    int id = int(v);

	    if (ix == 0 || *(im - xstride) != v)
	      faces.add_face(id, ix,iy,iz, 0, 0);  // add -x face

	    if (ix == xsm || *(im + xstride) != v)
	      faces.add_face(id, ix,iy,iz, 0, 1);  // add +x face

	    if (iy == 0 || *(im - ystride) != v)
	      faces.add_face(id, ix,iy,iz, 1, 0);  // add -y face

	    if (iy == ysm || *(im + ystride) != v)
	      faces.add_face(id, ix,iy,iz, 1, 1);  // add +y face

	    if (iz == 0 || *(im - zstride) != v)
	      faces.add_face(id, ix,iy,iz, 2, 0);  // add -z face

	    if (iz == zsm || *(im + zstride) != v)
	      faces.add_face(id, ix,iy,iz, 2, 1);  // add +z face
	}

  VertexMap vmap(xsize+1, ysize+1);
  faces.make_surfaces(surfaces, vmap);
}

// ----------------------------------------------------------------------------
//
template <class T>
void segment_group_surfaces(const Array<T> &image, const IArray &surface_ids,
			    bool surface_zero, Surfaces &surfaces)
{
  // axes x,y,z = 2,1,0
  int xsize = image.size(2), ysize = image.size(1), zsize = image.size(0);
  long xstride = image.stride(2), ystride = image.stride(1), zstride = image.stride(0);

  int *surf_ids = surface_ids.values();
  long id_stride = surface_ids.stride(0);
  int nids = surface_ids.size(0);

  // Create squares along each of 3 axes at each segment boundary point.
  FaceLists faces;
  T *im = image.values();
  int xsm = xsize - 1, ysm = ysize - 1, zsm = zsize - 1;
  for (int iz = 0 ; iz < zsize ; ++iz, im += zstride - ysize*ystride)
    for (int iy = 0 ; iy < ysize ; ++iy, im += ystride - xsize*xstride)
      for (int ix = 0 ; ix < xsize ; ++ix, im += xstride)
	{
	    T v = *im;
	    if (v < 0 || (int)v >= nids)
	      continue;
	    int id = surf_ids[(int)v * id_stride];
	    if (!surface_zero && id == 0)
	      continue;

	    if (ix == 0 || voxel_id(*(im - xstride), surf_ids, nids, id_stride) != id)
	      faces.add_face(id, ix,iy,iz, 0, 0);  // add -x face

	    if (ix == xsm || voxel_id(*(im + xstride), surf_ids, nids, id_stride) != id)
	      faces.add_face(id, ix,iy,iz, 0, 1);  // add +x face

	    if (iy == 0 || voxel_id(*(im - ystride), surf_ids, nids, id_stride) != id)
	      faces.add_face(id, ix,iy,iz, 1, 0);  // add -y face

	    if (iy == ysm || voxel_id(*(im + ystride), surf_ids, nids, id_stride) != id)
	      faces.add_face(id, ix,iy,iz, 1, 1);  // add +y face

	    if (iz == 0 || voxel_id(*(im - zstride), surf_ids, nids, id_stride) != id)
	      faces.add_face(id, ix,iy,iz, 2, 0);  // add -z face

	    if (iz == zsm || voxel_id(*(im + zstride), surf_ids, nids, id_stride) != id)
	      faces.add_face(id, ix,iy,iz, 2, 1);  // add +z face
	}

  VertexMap vmap(xsize+1, ysize+1);
  faces.make_surfaces(surfaces, vmap);
}

PyObject *python_surfaces(Surfaces &surfaces, bool delete_surfaces = true)
{
  PyObject *surfs = PyTuple_New(surfaces.size());
  for (size_t s = 0 ; s < surfaces.size() ; ++s)
    {
      Segment_Surface::Surface *surface = surfaces[s];
      PyObject *py_surf = python_tuple(PyLong_FromLong(surface->id()),
				       surface->python_vertex_array(),
				       surface->python_triangle_array());
      PyTuple_SetItem(surfs, s, py_surf);
      if (delete_surfaces)
	delete surface;
    }
  return surfs;
}

} // end of namespace Segment_Surface

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
segment_surface(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array image;
  double value;
  const char *kwlist[] = {"image", "value", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&d"),
				   (char **)kwlist,
				   parse_3d_array, &image,
				   &value))
    return NULL;
  
  PyObject *vt;
  try
    {
      Segment_Surface::Surface surface;
      Py_BEGIN_ALLOW_THREADS
	call_template_function(Segment_Surface::segment_surface, image.value_type(),
			       (image, value, surface));
      Py_END_ALLOW_THREADS
      vt = python_tuple(surface.python_vertex_array(),
			surface.python_triangle_array());
    }
  catch (std::bad_alloc&)
    {
      PyErr_Format(PyExc_MemoryError,
		   "segment_surface(): Out of memory, image size (%d,%d,%d)",
		   image.size(0), image.size(1), image.size(2));
      return NULL;
    }

  return vt;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
segment_surfaces(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array image;
  int surface_zero = 0;
  const char *kwlist[] = {"image", "zero", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&p"),
				   (char **)kwlist,
				   parse_3d_array, &image,
				   &surface_zero))
    return NULL;
  
  try
    {
      Segment_Surface::Surfaces surfaces;
      Py_BEGIN_ALLOW_THREADS
	call_template_function(Segment_Surface::segment_surfaces, image.value_type(),
			       (image, surface_zero, surfaces));
      Py_END_ALLOW_THREADS
      return python_surfaces(surfaces);
    }
  catch (std::bad_alloc&)
    {
      PyErr_Format(PyExc_MemoryError,
		   "segment_surfaces(): Out of memory, image size (%d,%d,%d)",
		   image.size(0), image.size(1), image.size(2));
      return NULL;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
segment_group_surfaces(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array image;
  IArray surface_ids;
  int surface_zero = 0;
  const char *kwlist[] = {"image", "surface_ids", "zero", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&p"),
				   (char **)kwlist,
				   parse_3d_array, &image,
				   parse_int_n_array, &surface_ids,
				   &surface_zero))
    return NULL;
  
  try
    {
      Segment_Surface::Surfaces surfaces;
      Py_BEGIN_ALLOW_THREADS
	call_template_function(Segment_Surface::segment_group_surfaces, image.value_type(),
			       (image, surface_ids, surface_zero, surfaces));
      Py_END_ALLOW_THREADS
      return python_surfaces(surfaces);
    }
  catch (std::bad_alloc&)
    {
      PyErr_Format(PyExc_MemoryError,
		   "segment_surfaces(): Out of memory, image size (%d,%d,%d)",
		   image.size(0), image.size(1), image.size(2));
      return NULL;
    }
}
