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
// This module loads the atomsearch shared library.
// By importing this module in Python it loads the library into the process
// so that other C++ Python modules that link against the library can find it.
//
#include <Python.h>			// use PyObject
#include <atomsearch/search.h>

// ----------------------------------------------------------------------------
//
static PyMethodDef atomic_lib_methods[] = {
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef atomic_lib_def =
{
	PyModuleDef_HEAD_INIT,
	"_atomic_lib",
	"Load atomic shared libraries.",
	-1,
	atomic_lib_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__load_libs()
{
  // do some stuff to force the library to link
  std::vector<atomstruct::Atom*> atoms;
  (void) atomsearch_search::AtomSearchTree(atoms, false, 5.0);
  return PyModule_Create(&atomic_lib_def);
}
