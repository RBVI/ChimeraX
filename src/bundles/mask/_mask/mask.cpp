// ----------------------------------------------------------------------------
//  Compute a 2-dimensional depth array from a list of triangles.  The
//  vertex y and x coordinates are indices into the depth array and the
//  z coordinate is the depth.  The depth array should be initialized to
//  a desired maximum depth before being passed to this routine.  If a
//  "beyond" array is passed it should be the same size as depth and
//  only depths beyond its values will be recorded in the depth array.
//  This can be used to get the second layer surface depth by passing in
//  a "beyond" array that is the depth calculated for the first layer.
//
//  Math needs to be done 64-bit to minimize round-off errors leading to
//  multiple nearly identical depths at single grid points where there is only
//  one surface point coincident with edge or vertex shared by multiple
//  triangles.
//
#include <Python.h>			// use PyObject

#include <math.h>			// use ceil(), floor()

#include <arrays/pythonarray.h>		// use parse_*_array()
#include <arrays/rcarray.h>		// use call_template_function()

// ---------------------------------------------------------------------------
//
inline int arg_min(double v0, double v1, double v2)
  { return (v0 < v1 ? (v0 < v2 ? 0 : 2) : (v1 < v2 ? 1 : 2)); }
inline int arg_max(double v0, double v1, double v2)
  { return (v0 > v1 ? (v0 > v2 ? 0 : 2) : (v1 > v2 ? 1 : 2)); }
inline int imin(int a, int b)
  { return (a < b ? a : b); }
inline int imax(int a, int b)
  { return (a > b ? a : b); }

// ---------------------------------------------------------------------------
// Compute z surface intercepts (depths) at a grid of xy positions for
// a surface defined by triangles.  Record only the depths greater
// than those given in the "beyond" array.  This allows multiple calls
// to compute depths for multiple surface layers.
//
// Implementation.  Z-axis grid lines that intercept triangle vertices
// or multiple points within a triangle present cases that can lead to
// erroneous multiple surface intercepts at depths differing by
// rounding precisions, or no surface intercept.  To handle these
// degenerate cases treat vertices with integral x or y coordinates as
// if they are displaced by an infinitesimal in the plus x or y
// direction.
//
static bool surface_z_depth(const FArray &varray, const IArray &tarray,
			    FArray &depth, IArray &tnum,
			    const FArray *beyond, const IArray *beyond_tnum,
			    int toffset)
{
  int64_t dsize[2] = {depth.size(1),depth.size(0)};		// x and y size

  bool set = false;

  int nt = tarray.size(0);
  const IArray tc = tarray.contiguous_array();
  const int *ta = tc.values();
  const FArray vc = varray.contiguous_array();
  const float *va = vc.values();
  float *da = depth.values();
  int *tn = tnum.values();
  FArray bc;
  const float *ba = (beyond ?
		     (bc = beyond->contiguous_array(), bc.values()) : NULL);
  IArray btc;
  const int *bt = (beyond_tnum ?
		   (btc = beyond_tnum->contiguous_array(),btc.values()) : NULL);

  double tv[3][3];
  for (int t = 0 ; t < nt ; ++t)
    {
      for (int v = 0 ; v < 3 ; ++v)
	{
	  int vi = ta[3*t+v];
	  for (int a = 0 ; a < 3 ; ++a)
	    tv[v][a] = va[3*vi+a];
	}
      // p denotes primary axis, s denotes secondary axis.
      const int p = 0, s = 1;
      int ipmin = arg_min(tv[0][p], tv[1][p], tv[2][p]);
      int ipmax = arg_max(tv[0][p], tv[1][p], tv[2][p]);
      if (ipmin == ipmax)
	continue;		// No intercepts possible.

      int ipmid = 3 - (ipmin + ipmax);
      double pmin = tv[ipmin][p], pmid = tv[ipmid][p], pmax = tv[ipmax][p];
      double pminc = ceil(pmin), pmaxf = floor(pmax);
      // Casts to double to avoid extra (80-bit) register precision.
      if (static_cast<double>(pminc) == static_cast<double>(pmin))
	pminc += 1;		// Offset coincident vertex.
      int pi0 = imax(0, (int)pminc);
      int pi1 = imin(dsize[p]-1, (int)pmaxf);
      for (int pi = pi0 ; pi <= pi1 ; ++pi)
	{
	  double fpa = (pi - pmin) / (pmax - pmin);
	  double sa = tv[ipmin][s]*(1-fpa) + tv[ipmax][s]*fpa;
	  double za = tv[ipmin][2]*(1-fpa) + tv[ipmax][2]*fpa;
	  double fsb, sb, zb;
	  if (pi < pmid)
	    {
	      fsb = (pi - pmin) / (pmid - pmin);
	      sb = tv[ipmin][s]*(1-fsb) + tv[ipmid][s]*fsb;
	      zb = tv[ipmin][2]*(1-fsb) + tv[ipmid][2]*fsb;
	    }
	  else
	    {
	      double xsep = pmax - pmid;
	      if (xsep == 0)
		fsb = 0;
	      else
		fsb = (pi - pmid) / xsep;
	      sb = tv[ipmid][s]*(1-fsb) + tv[ipmax][s]*fsb;
	      zb = tv[ipmid][2]*(1-fsb) + tv[ipmax][2]*fsb;
	    }
	  double smin, smax, zmin, zmax;
	  if (sa < sb)
	    { smin = sa; smax = sb; zmin = za; zmax = zb; }
	  else
	    { smin = sb; smax = sa; zmin = zb; zmax = za; }
	  double ssep = smax - smin, sminc = ceil(smin), smaxf = floor(smax);
	  // Casts to double to avoid extra (80-bit) register precision.
	  if (static_cast<double>(sminc) == static_cast<double>(smin))
	    sminc += 1;		// Offset coincident intercept.
	  int si0 = imax(0, (int)sminc);
	  int si1 = imin(dsize[s]-1, (int)smaxf);
	  for (int si = si0 ; si <= si1 ; ++si)
	    {
	      double fs = (ssep == 0 ? 0.5 : (si - smin) / ssep);
	      double z = zmin*(1-fs) + zmax*fs;
	      // Have to convert 64-bit z to 32-bit so same point
	      // does not appear beyond itself.
	      float z32 = static_cast<float>(z);
	      int di = (p == 0 ? si*dsize[0]+pi : pi*dsize[0]+si);
	      if (z32 < da[di])
		if (ba == NULL || z32 > ba[di] ||
		    (bt && z32 == ba[di] && t+toffset > bt[di]))
		  {
		    da[di] = z32;
		    tn[di] = t+toffset;
		    set = true;
		  }
	    }
	}
    }

  return set;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *py_surface_z_depth(PyObject *, PyObject *args,
					PyObject *keywds)
{
  int toffset = 0;
  FArray varray, depth, beyond;
  IArray tarray, tnum, beyond_tnum;
  const char *kwlist[] = {"vertices", "triangles", "depth", "triangle_number",
		    "beyond", "beyond_triangle_number",
		    "triangle_number_offset", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&|O&O&i"),
				   const_cast<char **>(kwlist),
				   parse_float_n3_array, &varray,
				   parse_int_n3_array, &tarray,
				   parse_writable_float_2d_array, &depth,
                                   parse_writable_int_2d_array, &tnum,
				   parse_float_2d_array, &beyond,
                                   parse_int_2d_array, &beyond_tnum,
                                   &toffset))
    return NULL;

  if (!depth.is_contiguous() || !tnum.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "Depth or triangle number array not contiguous.");
      return NULL;
    }

  FArray *beyondp = (beyond.dimension() == 0 ? NULL : &beyond);
  IArray *beyond_tnump = (beyond_tnum.dimension() == 0 ? NULL : &beyond_tnum);

  int s0 = depth.size(0), s1 = depth.size(1);
  if (tnum.size(0) != s0 || tnum.size(1) != s1 ||
      (beyondp && (beyond.size(0) != s0 || beyond.size(1) != s1)) ||
      (beyond_tnump && (beyond_tnum.size(0) != s0 || beyond_tnum.size(1) != s1))
      )
    {
      PyErr_SetString(PyExc_TypeError,
		      "Depth and triangle number array sizes don't match.");
      return NULL;
    }

  bool set = surface_z_depth(varray, tarray, depth, tnum,
			     beyondp, beyond_tnump, toffset);

  return python_bool(set);
}

// -----------------------------------------------------------------------------
//
template <class T>
void fill_slab(const FArray &depth, const FArray &depth2,
	       float mijk_to_dijk[3][4],
	       Reference_Counted_Array::Array<T> mvol,
	       float depth_limit)
{
  int ksize = mvol.size(0), jsize = mvol.size(1), isize = mvol.size(2);
  int djsize = depth.size(0), disize = depth.size(1);
  int ds0 = depth.stride(0), ds1 = depth.stride(1);
  int d2s0 = depth2.stride(0), d2s1 = depth2.stride(1);
  const float *da = depth.values(), *da2 = depth2.values();
  int ms0 = mvol.stride(0), ms1 = mvol.stride(1), ms2 = mvol.stride(2);
  T *mv = mvol.values();
  float (*t)[4]  = mijk_to_dijk;
  for (int k = 0 ; k < ksize ; ++k)
    for (int j = 0 ; j < jsize ; ++j)
      for (int i = 0 ; i < isize ; ++i)
	{
	  float di = t[0][0]*i + t[0][1]*j + t[0][2]*k + t[0][3];
	  float dj = t[1][0]*i + t[1][1]*j + t[1][2]*k + t[1][3];
	  float dk = t[2][0]*i + t[2][1]*j + t[2][2]*k + t[2][3];
	  if (di >= 0 && di < disize-1 && dj >= 0 && dj < djsize-1)
	    {
	      // Interpolate depths, nearest neighbor
	      // TODO: use linear interpolation.
	      int din = (int)(di + 0.5);
	      int djn = (int)(dj + 0.5);
	      float d1 = da[djn*ds0+din*ds1];
	      float d2 = da2[djn*d2s0+din*d2s1];
	      if (dk >= d1 && dk <= d2 &&
		  d1 <= depth_limit && d2 <= depth_limit)
		mv[k*ms0+j*ms1+i*ms2] = 1;
	    }
	}
}

// -----------------------------------------------------------------------------
//
extern "C" PyObject *py_fill_slab(PyObject *, PyObject *args, PyObject *keywds)
{
  float mijk_to_dijk[3][4], depth_limit;
  FArray depth, depth2;
  Numeric_Array mvol;
  const char *kwlist[] = {"depth", "depth2", "mijk_to_dijk", "mvol",
			  "depth_limit", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds,
				   const_cast<char *>("O&O&O&O&f"),
				   const_cast<char **>(kwlist),
				   parse_float_2d_array, &depth,
                                   parse_float_2d_array, &depth2,
				   parse_float_3x4_array, &mijk_to_dijk,
				   parse_3d_array, &mvol,
                                   &depth_limit))
    return NULL;

  if (depth.size(0) != depth2.size(0) || depth.size(1) != depth2.size(1))
    {
      PyErr_SetString(PyExc_TypeError, "Depth array sizes don't match");
      return NULL;
    }

  call_template_function(fill_slab, mvol.value_type(),
			 (depth, depth2, mijk_to_dijk, mvol, depth_limit));

  return python_none();
}

// -----------------------------------------------------------------------------
//
template <class T>
void pad_mask(Reference_Counted_Array::Array<T> mvol, int iter)
{
  int ksize = mvol.size(0), jsize = mvol.size(1), isize = mvol.size(2);
  int ms0 = mvol.stride(0), ms1 = mvol.stride(1), ms2 = mvol.stride(2);
  T *mv = mvol.values();
  T b = 1, bn = 2;
  for (int r = 0 ; r < iter ; ++r)
    {
      for (int k = 0, p = 0 ; k < ksize ; ++k, p += ms0 - jsize*ms1)
	for (int j = 0 ; j < jsize ; ++j, p += ms1 - isize*ms2)
	    for (int i = 0 ; i < isize ; ++i, p += ms2)
	      if (mv[p] == b)
		{
		  if (i > 0 && mv[p-ms2] == 0) mv[p-ms2] = bn;
		  if (i+1 < isize && mv[p+ms2] == 0) mv[p+ms2] = bn;
		  if (j > 0 && mv[p-ms1] == 0) mv[p-ms1] = bn;
		  if (j+1 < jsize && mv[p+ms1] == 0) mv[p+ms1] = bn;
		  if (k > 0 && mv[p-ms0] == 0) mv[p-ms0] = bn;
		  if (k+1 < ksize && mv[p+ms0] == 0) mv[p+ms0] = bn;
		}
      b += 1;
      bn += 1;
    }

  for (int k = 0, p = 0 ; k < ksize ; ++k, p += ms0 - jsize*ms1)
    for (int j = 0 ; j < jsize ; ++j, p += ms1 - isize*ms2)
      for (int i = 0 ; i < isize ; ++i, p += ms2)
	if (mv[p] >= 1)
	      mv[p] = 1;
}

// -----------------------------------------------------------------------------
//
extern "C" PyObject *py_pad_mask(PyObject *, PyObject *args, PyObject *keywds)
{
  int iter = 1;
  Numeric_Array mvol;
  const char *kwlist[] = {"volume", "iterations", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&|i"),
				   const_cast<char **>(kwlist),
				   parse_3d_array, &mvol,
                                   &iter))
    return NULL;

  call_template_function(pad_mask, mvol.value_type(), (mvol, iter));

  return python_none();
}

// ----------------------------------------------------------------------------
//
static struct PyMethodDef mask_methods[] =
{
  /* name, address, '1' = tuple arg-lists */
  {const_cast<char*>("surface_z_depth"), (PyCFunction)py_surface_z_depth,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("fill_slab"), (PyCFunction)py_fill_slab,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {const_cast<char*>("pad_mask"), (PyCFunction)py_pad_mask,
   METH_VARARGS|METH_KEYWORDS, NULL},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_mask",
        NULL,
        -1,
        mask_methods,
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__mask(void)
{
    return PyModule_Create(&moduledef);
}
