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

#include <arrays/pythonarray.h>		// use parse_uint8_n_array()
#include <arrays/rcarray.h>		// use CArray

// ----------------------------------------------------------------------------
//
static int count_value(unsigned char *a, long n, long stride, unsigned char v)
{
  int c = 0;
  for (long i = 0 ; i < n ; ++i, a += stride)
    if (*a == v)
      c += 1;
  return c;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
count_value(PyObject *, PyObject *args, PyObject *keywds)
{
  int v;
  CArray a;
  const char *kwlist[] = {"array", "value", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i"),
				   (char **)kwlist,
				   parse_uint8_n_array, &a,
				   &v))
    return NULL;

  int c = count_value((unsigned char *)a.values(), a.size(), a.stride(0), (unsigned char)v);
  return PyLong_FromLong(c);
}
