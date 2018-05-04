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
//
#include <Python.h>			// use PyObject	
#include <math.h>			// use fmod()

#include <arrays/pythonarray.h>		// use array_from_python(), ...
#include <arrays/rcarray.h>		// use Array<T>, Numeric_Array
using Reference_Counted_Array::Numeric_Array;

namespace Map_Cpp
{

// ----------------------------------------------------------------------------
//
inline float wrap(float f, int i)
{
  float frac = fmod(f, (float)i);
  if (frac < 0)
    frac += i;
  return frac;
}

// ----------------------------------------------------------------------------
//
template <class T>
bool interpolate(const T *ia,
		 int isz, int jsz, int ksz, long ist, long jst, long kst,
		 float i, float j, float k, float *v)
{
  if (i < 0 || j < 0 || k < 0 || i > isz-1 || j > jsz-1 || k > ksz-1)
    return false;

  int bi = (int)i, bj = (int)j, bk = (int)k;
  if (bi == isz-1) bi -= 1;
  if (bj == jsz-1) bj -= 1;
  if (bk == ksz-1) bk -= 1;
  float fi1 = i-bi, fj1 = j-bj, fk1 = k-bk;
  float fi0 = 1 - fi1, fj0 = 1 - fj1, fk0 = 1 - fk1;
  const T *c = ia + bi*ist + bj*jst + bk*kst;
  *v = (fk0*(fj0*(fi0 * c[0] + fi1 * c[ist]) +
	     fj1*(fi0 * c[jst] + fi1 * c[ist+jst])) +
	fk1*(fj0*(fi0 * c[kst] + fi1 * c[ist+kst]) +
	     fj1*(fi0 * c[jst+kst] + fi1 * c[ist+jst+kst])));

  return true;	
}

// ----------------------------------------------------------------------------
//
template <class T>
void extend_map(const Reference_Counted_Array::Array<T> &in, int cell_size[3],
		const FArray &syms, FArray &out, float out_to_in_tf[3][4],
		long *nmiss, float *dmax)
{
  int kinsz = in.size(0), jinsz = in.size(1), iinsz = in.size(2);
  int kinst = in.stride(0), jinst = in.stride(1), iinst = in.stride(2);
  const T *ia = in.values();

  int ksz = out.size(0), jsz = out.size(1), isz = out.size(2);
  long kst = out.stride(0), jst = out.stride(1), ist = out.stride(2);
  float *oa = out.values();

  int csi = cell_size[0], csj = cell_size[1], csk = cell_size[2];

  int nsym = syms.size(0);
  float *sa = syms.values();

  float *oi = &out_to_in_tf[0][0];

  *nmiss = 0;
  *dmax = 0;

  for (int k = 0 ; k < ksz ; ++k)
    for (int j = 0 ; j < jsz ; ++j)
      for (int i = 0 ; i < isz ; ++i)
	{
	  float ini = oi[0]*i + oi[1]*j + oi[2]*k + oi[3];
	  float inj = oi[4]*i + oi[5]*j + oi[6]*k + oi[7];
	  float ink = oi[8]*i + oi[9]*j + oi[10]*k + oi[11];
	  int inside = 0;
	  float vsum = 0, vmin = 0, vmax = 0;
	  for (int s = 0 ; s < nsym ; ++s)
	    {
	      float *sym = &sa[s*12];
	      float si = sym[0]*ini + sym[1]*inj + sym[2]*ink + sym[3];
	      float sj = sym[4]*ini + sym[5]*inj + sym[6]*ink + sym[7];
	      float sk = sym[8]*ini + sym[9]*inj + sym[10]*ink + sym[11];
	      float usi = wrap(si, csi);
	      float usj = wrap(sj, csj);
	      float usk = wrap(sk, csk);
	      float v;
	      if (interpolate(ia, iinsz, jinsz, kinsz, iinst, jinst, kinst,
			      usi, usj, usk, &v))
		{
		  vsum += v;
		  if (inside == 0)
		    vmin = vmax = v;
		  else if (v > vmax)
		    vmax = v;
		  else if (v < vmin)
		    vmin = v;
		  inside += 1;
		}
	    }
	  float vijk;
	  if (inside == 0)
	    {
	      *nmiss += 1;
	      vijk = 0;
	    }
	  else
	    {
	      vijk = vsum / inside;
	      float d = vmax - vmin;
	      if (d > *dmax)
		*dmax = d;
	    }
	  long oi = k*kst + j*jst + i*ist;
	  oa[oi] = vijk;
	}		 
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
extend_crystal_map(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array in;
  int csize[3];
  FArray syms, out;
  float oitf[3][4];
  const char *kwlist[] = {"in_array", "ijk_cell_size", "ijk_symmetries",
			  "out_array", "out_ijk_to_in_ijk_transform", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&O&"),
				   (char **)kwlist,
				   parse_3d_array, &in,
				   parse_int_3_array, &csize,
				   parse_float_array, &syms,
				   parse_writable_float_3d_array, &out,
				   parse_float_3x4_array, &oitf))
    return NULL;

  if (syms.dimension() != 3)
    {
      PyErr_SetString(PyExc_TypeError,
		      "ijk_symmetries must be 3 dimensional");
      return NULL;
    }
  if (syms.size(1) != 3 || syms.size(2) != 4)
    {
      PyErr_SetString(PyExc_TypeError,
		      "ijk_symmetries must be array of 3 by 4 matrices");
      return NULL;
    }
  if (!syms.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "Require contiguous ijk_symmetries array");
      return NULL;
    }

  long nmiss;
  float dmax;
  call_template_function(extend_map, in.value_type(),
  			 (in, csize, syms, out, oitf, &nmiss, &dmax));

  PyObject *py_nmiss = PyLong_FromLong(nmiss);
  PyObject *py_dmax = PyFloat_FromDouble(dmax);
  PyObject *ret = python_tuple(py_nmiss, py_dmax);
  return ret;
}

}	// end of namespace Map_Cpp
