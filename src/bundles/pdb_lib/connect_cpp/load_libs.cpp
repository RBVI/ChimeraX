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
// This module loads the pdbconnect shared library.
// By importing this module in Python it loads the library into the process
// so that other C++ Python modules that link against the library can find it.
//
#include <Python.h>			// use PyObject
#include <pdb/connect.h>

// ----------------------------------------------------------------------------
//
static PyMethodDef pdb_lib_methods[] = {
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef pdb_lib_def =
{
	PyModuleDef_HEAD_INIT,
	"_load_libs",
	"Load pdb shared libraries.",
	-1,
	pdb_lib_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__load_libs()
{
  // do some stuff to force the library to link
  atomstruct::Structure s;
  pdb_connect::find_and_add_metal_coordination_bonds(&s);
  return PyModule_Create(&pdb_lib_def);
}
