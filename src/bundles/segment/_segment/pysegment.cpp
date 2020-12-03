// ----------------------------------------------------------------------------
// Python wrapper for _segment module.
//
#include <Python.h>			// use PyObject

#include <stdexcept>			// use std::runtime_error
#include <iostream>
#include <string.h>			// use memset()

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray, IArray

#include "bin.h"			// use bin_sums()
#include "region_map.h"			// use region_grid_indices(), Index
#include "watershed.h"			// Use watershed_regions()

using namespace Reference_Counted_Array;

namespace Segment_Map
{

// ----------------------------------------------------------------------------
//
static bool parse_map(PyObject *py_data, Numeric_Array *data)
{
  return array_from_python(py_data, 3, data);
}

// ----------------------------------------------------------------------------
//
static bool parse_region_map(PyObject *py_region_map, Array<unsigned int> &region_map)
{
  Numeric_Array a;
  if (!array_from_python(py_region_map, 3, &a, false))
    return false;
  if (a.value_type() == Numeric_Array::Unsigned_Int ||
      (sizeof(unsigned long) == sizeof(unsigned int) &&
       a.value_type() == Numeric_Array::Unsigned_Long_Int))
    {
      region_map = a;
      return true;
    }

  PyErr_SetString(PyExc_TypeError, "Array type is not uint32");
  return false;
}

// ----------------------------------------------------------------------------
// Returns number of regions found.  Region_Map must be contiguous array.
//
template <class T>
void watershed_reg(const Array<T> &data, float threshold, Index *region_map, Index *rcount)
{
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  *rcount = watershed_regions(d, data.sizes(), threshold, region_map);
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *watershed_regions(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_data, *py_region_map;
  float threshold;
  const char *kwlist[] = {"data", "threshold", "region_map", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OfO"),
				   (char **)kwlist, &py_data, &threshold,
				   &py_region_map))
    return NULL;

  
  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;
  if (!region_map.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "region_map array must be contiguous");
      return NULL;
    }

  Index rcount;
  call_template_function(watershed_reg, data.value_type(),
  			 (data, threshold, region_map.values(), &rcount));

  return PyLong_FromLong(rcount);
}

// ----------------------------------------------------------------------------
//
static PyObject *region_map_region_indices(const Array<Index> &region_map)
{
  Array<Index> mc = region_map.contiguous_array();
  Index *m = mc.values();

  // Count regions.
  Index c = largest_value(m, region_map.sizes());

  // Count grid points in each region.
  Index *rc = new Index[c+1];
  region_sizes(m, region_map.sizes(), c, rc);

  Index nr = 0;
  for (Index r = 1 ; r <= c ; ++r)
    if (rc[r] > 0)
      nr += 1;

  // Make a numpy array for grid indices for each region,
  // excluding the index 0 region.
  PyObject *t = PyTuple_New(nr);
  int **gi = new int*[c+1];
  for (Index r = 1, ti = 0 ; r <= c ; ++r)
    if (rc[r] > 0)
      {
	PyTuple_SetItem(t, ti, python_int_array(rc[r], 3, &gi[r]));
	ti += 1;
      }
    else
      gi[r] = NULL;
  gi[0] = NULL;
  delete [] rc;

  // Record grid indices.
  region_grid_indices(m, region_map.sizes(), gi);
  delete [] gi;

  return t;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_index_lists(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map;
  const char *kwlist[] = {"region_map", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &py_region_map))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  PyObject *mi = region_map_region_indices(region_map);
  return mi;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_contacts(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map;
  const char *kwlist[] = {"region_map", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &py_region_map))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  Array<Index> mc = region_map.contiguous_array();
  Index *m = mc.values();

  Contacts contacts;
  region_contacts(m, region_map.sizes(), contacts);

  size_t nc = contacts.size();
  int *con;
  PyObject *cpy = python_int_array(nc, 3, &con);
  for (size_t c = 0 ; c < nc ; ++c)
    {
      Contact &cc = contacts[c];
      con[3*c] = cc.region1;
      con[3*c+1] = cc.region2;
      con[3*c+2] = cc.ncontact;
    }

  return cpy;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_bounds(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map;
  const char *kwlist[] = {"region_map", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &py_region_map))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  Array<Index> mc = region_map.contiguous_array();
  Index *m = mc.values();

  Index rmax = largest_value(m, region_map.sizes());
  int *bounds;
  PyObject *bpy = python_int_array(rmax+1, 7, &bounds);
  region_bounds(m, region_map.sizes(), rmax, bounds);

  return bpy;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_point_count(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map;
  int rid;
  const char *kwlist[] = {"region_map", "region_id", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Oi"),
				   (char **)kwlist, &py_region_map, &rid))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  unsigned int c = region_point_count(region_map.values(), region_map.sizes(),
				      region_map.strides(), (Index) rid);

  return PyLong_FromLong(c);
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_points(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map;
  int rid;
  const char *kwlist[] = {"region_map", "region_id", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Oi"),
				   (char **)kwlist, &py_region_map, &rid))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  unsigned int pcount = region_point_count(region_map.values(), region_map.sizes(),
					   region_map.strides(), (Index) rid);
  int *points;
  PyObject *pts = python_int_array(pcount, 3, &points);
  
  region_points(region_map.values(), region_map.sizes(), region_map.strides(),
		(Index) rid, points);

  return pts;
}

// ----------------------------------------------------------------------------
//
template <class T>
void interface_val(Array<unsigned int> &region_map, const Array<T> &data,
		   Contacts &contacts)
{
  Array<Index> mc = region_map.contiguous_array();
  Index *m = mc.values();
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  interface_values(m, region_map.sizes(), d, contacts);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interface_values(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map, *py_data;
  const char *kwlist[] = {"region_map", "data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OO"),
				   (char **) kwlist, &py_region_map, &py_data))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  Contacts contacts;
  call_template_function(interface_val, data.value_type(),
  			 (region_map, data, contacts));

  size_t nc = contacts.size();
  int *con;
  PyObject *cpy = python_int_array(nc, 3, &con);
  float *cf;
  PyObject *cfpy = python_float_array(nc, 2, &cf);
  for (size_t c = 0 ; c < nc ; ++c)
    {
      Contact &cc = contacts[c];
      con[3*c] = cc.region1;
      con[3*c+1] = cc.region2;
      con[3*c+2] = cc.ncontact;
      cf[2*c] = cc.data_max;
      cf[2*c+1] = cc.data_sum;
    }

  PyObject *t = python_tuple(cpy, cfpy);
  return t;
}

// ----------------------------------------------------------------------------
//
template <class T>
void region_max(Array<unsigned int> &region_map, const Array<T> &data, Index nmax,
		int *max_points, float *max_values)
{
  Array<Index> mc = region_map.contiguous_array();
  Index *m = mc.values();
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  region_maxima(m, region_map.sizes(), d, nmax, max_points, max_values);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *region_maxima(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_region_map, *py_data;
  const char *kwlist[] = {"region_map", "data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OO"),
				   (char **) kwlist, &py_region_map, &py_data))
    return NULL;

  Array<unsigned int> region_map;
  if (!parse_region_map(py_region_map, region_map))
    return NULL;

  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  Array<Index> mc = region_map.contiguous_array();
  Index *m = mc.values();
  Index nmax = largest_value(m, region_map.sizes());
  int *max_points;
  PyObject *pts = python_int_array(nmax, 3, &max_points);
  float *max_values;
  PyObject *vals = python_float_array(nmax, &max_values);

  call_template_function(region_max, data.value_type(),
  			 (mc, data, nmax, max_points, max_values));

  PyObject *t = python_tuple(pts, vals);
  return t;
}

// ----------------------------------------------------------------------------
// Returns number of regions found.  If data array is not contiguous it will be copied.
//
template <class T>
void find_local_max(const Array<T> &data, int *positions, int npos)
{
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  find_local_maxima(d, data.sizes(), positions, npos);
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *find_local_maxima(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_data;
  IArray pos;
  const char *kwlist[] = {"data", "start_positions", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OO&"),
				   (char **)kwlist, &py_data,
				   parse_int_n3_array, &pos))
    return NULL;

  if (!pos.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "start_position array not contiguous");
      return NULL;
    }

  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  call_template_function(find_local_max, data.value_type(),
  			 (data, pos.values(), pos.size(0)));

  return python_none();
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *crosssection_midpoints(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points;
  float axis[3], b0, bsize;
  int bcount;
  
  const char *kwlist[] = {"points", "axis", "bin_start", "bin_size", "bin_count", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&ffi"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_3_array, &axis[0],
				   &b0, &bsize, &bcount))
    return NULL;

  if (!points.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "points array not contiguous");
      return NULL;
    }

  int *bcounts;
  PyObject *pybc = python_int_array(bcount, &bcounts);
  memset(bcounts, 0, bcount * sizeof(int));
  float *bsums;
  PyObject *pybs = python_float_array(bcount, 3, &bsums);
  memset(bsums, 0, bcount * 3 * sizeof(float));

  bin_sums(points.values(), points.size(0), axis, b0, bsize, bcount,
	   bsums, bcounts);

  PyObject *sc = python_tuple(pybs, pybc);
  return sc;
}

}	// end of namespace Segment_Cpp
