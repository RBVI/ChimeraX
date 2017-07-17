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

#include <logger/logger.h>

#include <atomstruct/Atom.h>
#include <atomstruct/Coord.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Structure.h>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

//
// compute_ss
//    Compute Kabsch & Sander DSSP secondary structure
//
// This is an implementation of
//
//    Dictionary of Protein Secondary Structure:
//    Pattern Recognition of Hydrogen-Bonded and
//    Geometrical Features
//    Wolfgang Kabsch and Christian Sander
//    Biopolymers, Vol. 22, 2577-2637 (1983)
//
//    The Python function takes five mandatory arguments:
//        pointer to a Structure (int)
//        hbond energy cutoff (float)
//        minimum helix length (int)
//        minimum strand length (int)
//        whether to report summary of dssp computation [ladders, etc.] (bool)
//
extern "C" {

static
PyObject *
compute_ss(PyObject *, PyObject *args)
{
    PyObject* ptr;
    double energy_cutoff;
    int min_helix_length, min_strand_length;
    int report;
    if (!PyArg_ParseTuple(args, PY_STUPID "Odiip", &ptr, &energy_cutoff,
        &min_helix_length, &min_strand_length, &report))
        return nullptr;
    // convert first arg to Structure*
    if (!PyLong_Check(ptr)) {
        PyErr_SetString(PyExc_TypeError, "First arg not an int (structure pointer)");
        return nullptr;
    }
	using atomstruct::Structure;
    Structure* mol = static_cast<Structure*>(PyLong_AsVoidPtr(ptr));
    try {
        mol->compute_secondary_structure(static_cast<float>(energy_cutoff), min_helix_length,
			min_strand_length, static_cast<bool>(report));
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
		return nullptr;
    }
	Py_INCREF(Py_None);
    return Py_None;
}

}

static const char* docstr_compute_ss =
"compute_ss\n"
"Compute/assign Kabsch & Sander DSSP secondary structure\n"
"\n"
"The function takes five arguments:\n"
"    mol_ptr        pointer to Structure\n"
"    cutoff        hbond energy cutoff\n"
"    min_h_len    minimum helix length\n"
"    min_s_len    minimum strand length\n"
"    do_report    whether to log computed values\n";

static PyMethodDef dssp_methods[] = {
    { PY_STUPID "compute_ss", compute_ss,    METH_VARARGS, PY_STUPID docstr_compute_ss },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef dssp_def =
{
    PyModuleDef_HEAD_INIT,
    "_dssp",
    "Compute secondary structure via Kabsch & Sander DSSP method",
    -1,
    dssp_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC
PyInit__dssp()
{
    return PyModule_Create(&dssp_def);
}
