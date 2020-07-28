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
// This module loads the libarrays.so shared library.
// By importing this module in Python it loads libarrays.so into the process
// so that other C++ Python modules that link against libarrays.so can find it.
//
#include <Python.h>			// use PyObject

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use FArray, IArray

// ----------------------------------------------------------------------------
//
static PyMethodDef arrays_methods[] = {
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef arrays_def =
{
	PyModuleDef_HEAD_INIT,
	"_arrays",
	"Load libarrays shared library.",
	-1,
	arrays_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__arrays()
{
  python_none();  // Call libarrays.so so it must be linked.
  return PyModule_Create(&arrays_def);
}
