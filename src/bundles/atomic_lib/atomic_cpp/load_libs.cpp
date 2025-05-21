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
// This module loads the atomic shared libraries.
// By importing this module in Python it loads the libraries into the process
// so that other C++ Python modules that link against those libraries can find them.
//
#include <Python.h>			// use PyObject
#include <atomstruct/Structure.h>

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
  // do some stuff to force the libraries to link
  auto s = atomstruct::Structure();
  (void) s.new_atom("X", element::Element::get_element("C"));
  return PyModule_Create(&atomic_lib_def);
}
