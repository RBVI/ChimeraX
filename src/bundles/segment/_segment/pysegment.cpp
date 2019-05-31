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
#include "mask.h"			// use region_grid_indices(), Index
#include "watershed.h"			// Use watershed_regions()

using namespace Reference_Counted_Array;
using namespace Segmentation_Calculation;

// ----------------------------------------------------------------------------
//
static bool parse_map(PyObject *py_data, Numeric_Array *data)
{
  return array_from_python(py_data, 3, data);
}

// ----------------------------------------------------------------------------
//
static bool parse_mask(PyObject *py_mask, Array<unsigned int> &mask)
{
  Numeric_Array a;
  if (!array_from_python(py_mask, 3, &a, false))
    return false;
  if (a.value_type() == Numeric_Array::Unsigned_Int ||
      (sizeof(unsigned long) == sizeof(unsigned int) &&
       a.value_type() == Numeric_Array::Unsigned_Long_Int))
    {
      mask = a;
      return true;
    }

  PyErr_SetString(PyExc_TypeError, "Array type is not uint32");
  return false;
}

// ----------------------------------------------------------------------------
// Returns number of regions found.  Mask must be contiguous array.
//
template <class T>
void watershed_reg(const Array<T> &data, float threshold, Index *mask, Index *rcount)
{
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  *rcount = watershed_regions(d, data.sizes(), threshold, mask);
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *watershed_regions_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_data, *py_mask;
  float threshold;
  const char *kwlist[] = {"data", "threshold", "mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OfO"),
				   (char **)kwlist, &py_data, &threshold,
				   &py_mask))
    return NULL;

  
  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;
  if (!mask.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "mask array must be contiguous");
      return NULL;
    }

  Index rcount;
  call_template_function(watershed_reg, data.value_type(),
  			 (data, threshold, mask.values(), &rcount));

  return PyLong_FromLong(rcount);
}

// ----------------------------------------------------------------------------
//
static PyObject *mask_region_indices(const Array<Index> &mask)
{
  Array<Index> mc = mask.contiguous_array();
  Index *m = mc.values();

  // Count regions.
  Index c = largest_value(m, mask.sizes());

  // Count grid points in each region.
  Index *rc = new Index[c+1];
  region_sizes(m, mask.sizes(), c, rc);

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
  region_grid_indices(m, mask.sizes(), gi);
  delete [] gi;

  return t;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_index_lists_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask;
  const char *kwlist[] = {"mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &py_mask))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  PyObject *mi = mask_region_indices(mask);
  return mi;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_contacts_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask;
  const char *kwlist[] = {"mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &py_mask))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  Array<Index> mc = mask.contiguous_array();
  Index *m = mc.values();

  Contacts contacts;
  region_contacts(m, mask.sizes(), contacts);

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
extern "C"  PyObject *region_bounds_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask;
  const char *kwlist[] = {"mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &py_mask))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  Array<Index> mc = mask.contiguous_array();
  Index *m = mc.values();

  Index rmax = largest_value(m, mask.sizes());
  int *bounds;
  PyObject *bpy = python_int_array(rmax+1, 7, &bounds);
  region_bounds(m, mask.sizes(), rmax, bounds);

  return bpy;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_point_count_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask;
  int rid;
  const char *kwlist[] = {"mask", "region_id", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Oi"),
				   (char **)kwlist, &py_mask, &rid))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  unsigned int c = region_point_count(mask.values(), mask.sizes(),
				      mask.strides(), (Index) rid);

  return PyLong_FromLong(c);
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *region_points_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask;
  int rid;
  const char *kwlist[] = {"mask", "region_id", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("Oi"),
				   (char **)kwlist, &py_mask, &rid))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  unsigned int pcount = region_point_count(mask.values(), mask.sizes(),
					   mask.strides(), (Index) rid);
  int *points;
  PyObject *pts = python_int_array(pcount, 3, &points);
  
  region_points(mask.values(), mask.sizes(), mask.strides(),
		(Index) rid, points);

  return pts;
}

// ----------------------------------------------------------------------------
//
template <class T>
void interface_val(Array<unsigned int> &mask, const Array<T> &data,
		   Contacts &contacts)
{
  Array<Index> mc = mask.contiguous_array();
  Index *m = mc.values();
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  interface_values(m, mask.sizes(), d, contacts);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *interface_values_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask, *py_data;
  const char *kwlist[] = {"mask", "data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OO"),
				   (char **) kwlist, &py_mask, &py_data))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  Contacts contacts;
  call_template_function(interface_val, data.value_type(),
  			 (mask, data, contacts));

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
void region_max(Array<unsigned int> &mask, const Array<T> &data, Index nmax,
		int *max_points, float *max_values)
{
  Array<Index> mc = mask.contiguous_array();
  Index *m = mc.values();
  Array<T> dc = data.contiguous_array();
  T *d = dc.values();
  region_maxima(m, mask.sizes(), d, nmax, max_points, max_values);
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *region_maxima_py(PyObject *, PyObject *args, PyObject *keywds)
{
  PyObject *py_mask, *py_data;
  const char *kwlist[] = {"mask", "data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("OO"),
				   (char **) kwlist, &py_mask, &py_data))
    return NULL;

  Array<unsigned int> mask;
  if (!parse_mask(py_mask, mask))
    return NULL;

  Numeric_Array data;
  if (!parse_map(py_data, &data))
    return NULL;

  Array<Index> mc = mask.contiguous_array();
  Index *m = mc.values();
  Index nmax = largest_value(m, mask.sizes());
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
// Returns number of regions found.  Mask must be contiguous array.
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
extern "C"  PyObject *find_local_maxima_py(PyObject *, PyObject *args, PyObject *keywds)
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

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
//
extern "C"  PyObject *bin_sums_py(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray points;
  float v[3], b0, bsize;
  int bcount;
  
  const char *kwlist[] = {"points", "v", "b0", "bsize", "bcount", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&ffi"),
				   (char **)kwlist,
				   parse_float_n3_array, &points,
				   parse_float_3_array, &v[0],
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

  bin_sums(points.values(), points.size(0), v, b0, bsize, bcount,
	   bsums, bcounts);

  PyObject *sc = python_tuple(pybs, pybc);
  return sc;
}

// ----------------------------------------------------------------------------
//
static struct PyMethodDef segment_methods[] =
{
  /* name, address, '1' = tuple arg-lists */
  {"watershed_regions", (PyCFunction)watershed_regions_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"region_index_lists", (PyCFunction)region_index_lists_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"region_contacts", (PyCFunction)region_contacts_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"region_bounds", (PyCFunction)region_bounds_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"region_point_count", (PyCFunction)region_point_count_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"region_points", (PyCFunction)region_points_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"region_maxima", (PyCFunction)region_maxima_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"interface_values", (PyCFunction)interface_values_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"find_local_maxima", (PyCFunction)find_local_maxima_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {"bin_sums", (PyCFunction)bin_sums_py,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef segment_def =
{
	PyModuleDef_HEAD_INIT,
	"_segment",
	"watershed segmentation",
	-1,
	segment_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__segment()
{
	return PyModule_Create(&segment_def);
}
