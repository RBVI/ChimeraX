// vi: set expandtab ts=4 sw=4

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

#include <Python.h>
#include <algorithm>
#include <list>
#include <map>
#include <string>

#include <atomstruct/Atom.h>
#include <atomstruct/Coord.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Structure.h>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

//
// connect_structure
//    Add bonds to structure based on inter-atomic distances
//    and missing-structure pseudobonds as appropriate
//
void
connect_structure(AtomStructure* s)
{
	
}

extern "C" {

static
PyObject *
py_connect_structure(PyObject *, PyObject *args)
{
    PyObject* ptr;
    if (!PyArg_ParseTuple(args, PY_STUPID "O", &ptr))
        return nullptr;
    // convert first arg to Structure*
    if (!PyLong_Check(ptr)) {
        PyErr_SetString(PyExc_TypeError, "First arg not an int (structure pointer)");
        return nullptr;
    }
	using atomstruct::AtomicStructure;
    AtomicStructure* mol = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(ptr));
    try {
		connect_structure(mol);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
		return nullptr;
    }
	Py_INCREF(Py_None);
    return Py_None;
}

}

static const char* docstr_connect_structure = "connect_structure(AtomicStructure)";

static PyMethodDef connect_structure_methods[] = {
    { PY_STUPID "connect_structure", py_connect_structure,    METH_VARARGS, PY_STUPID docstr_connect_structure },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef connect_structure_def =
{
    PyModuleDef_HEAD_INIT,
    "_cs",
    "Add bonds to structure based on atom distances",
    -1,
    connect_structure_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC
PyInit__cs()
{
    return PyModule_Create(&connect_structure_def);
}
