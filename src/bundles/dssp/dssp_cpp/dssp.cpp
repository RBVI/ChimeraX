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
#include <memory>
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

using atomstruct::CompSSInfo;
using atomstruct::Residue;
using atomstruct::Structure;

inline PyObject*
make_residue(Residue *r)
{
    return r->py_instance(true);
}

static
PyObject *
compute_ss(PyObject *, PyObject *args, PyObject* keywds)
{
    PyObject* ptr;
    double energy_cutoff = -0.5;
    int min_helix_length = 3, min_strand_length = 3;
    int report = false;
    int return_values = false;
    std::unique_ptr<CompSSInfo> ss_data;
    static const char* kwlist[] = {
        "", "energy_cutoff", "min_helix_len", "min_strand_len", "report", "return_values", nullptr };
    if (!PyArg_ParseTupleAndKeywords(
             args, keywds, PY_STUPID "O|diipp", (char**) kwlist,
             &ptr, &energy_cutoff, &min_helix_length, &min_strand_length,
             &report, &return_values))
        return nullptr;
    // convert first arg to Structure*
    if (!PyLong_Check(ptr)) {
        PyErr_SetString(PyExc_TypeError, "First arg not an int (structure pointer)");
        return nullptr;
    }
    Structure* mol = static_cast<Structure*>(PyLong_AsVoidPtr(ptr));
    if (return_values)
        ss_data = std::unique_ptr<CompSSInfo>(new CompSSInfo);
    try {
        mol->compute_secondary_structure(static_cast<float>(energy_cutoff), min_helix_length,
                min_strand_length, static_cast<bool>(report), ss_data.get());
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
    }
    if (!return_values) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyObject* data = PyDict_New();
        PyObject* strands = PyList_New(ss_data->strands.size());
        for (auto i = 0; i < ss_data->strands.size(); ++i) {
            auto &strand = ss_data->strands.at(i);
            auto pair = PyTuple_New(2);
            PyTuple_SET_ITEM(pair, 0, make_residue(strand.first));
            PyTuple_SET_ITEM(pair, 1, make_residue(strand.second));
            PyList_SET_ITEM(strands, i, pair);
        }
        PyDict_SetItemString(data, "strands", strands);
        Py_DECREF(strands);

        PyObject* sheets = PyList_New(ss_data->sheets.size());
        for (auto i = 0; i < ss_data->sheets.size(); ++i) {
            auto &sheet = ss_data->sheets.at(i);
            auto indices = PySet_New(NULL);
            for (auto j: sheet) {
                auto index = PyLong_FromLong(j);
                PySet_Add(indices, index);
                Py_DECREF(index);
            }
            PyList_SET_ITEM(sheets, i, indices);
        }
        PyDict_SetItemString(data, "sheets", sheets);
        Py_DECREF(sheets);

        PyObject* parallels = PyDict_New();
        for (auto &item: ss_data->strands_parallel) {
            auto pair = PyTuple_New(2);
            PyTuple_SET_ITEM(pair, 0, PyLong_FromLong(item.first.first));
            PyTuple_SET_ITEM(pair, 1, PyLong_FromLong(item.first.second));
            auto parallel = PyBool_FromLong(item.second);
            PyDict_SetItem(parallels, pair, parallel);
            Py_DECREF(pair);
            Py_DECREF(parallel);
        }
        PyDict_SetItemString(data, "strands_parallel", parallels);
        Py_DECREF(parallels);

        PyObject* helices = PyList_New(ss_data->helix_info.size());
        for (auto i = 0; i < ss_data->helix_info.size(); ++i) {
            auto &helix = ss_data->helix_info.at(i);
            auto pair = PyTuple_New(2);
            PyTuple_SET_ITEM(pair, 0, make_residue(helix.first.first));
            PyTuple_SET_ITEM(pair, 1, make_residue(helix.first.second));
            auto type = PyLong_FromLong(helix.second);
            auto datum = PyTuple_New(2);
            PyTuple_SET_ITEM(datum, 0, pair);
            PyTuple_SET_ITEM(datum, 1, type);
            PyList_SET_ITEM(helices, i, datum);
        }
        PyDict_SetItemString(data, "helex_info", helices);
        Py_DECREF(helices);
        return data;
    }
}

}

static const char* docstr_compute_ss =
"compute_ss\n"
"Compute/assign Kabsch & Sander DSSP secondary structure\n"
"\n"
"The arguments are:\n"
"    mol_ptr        pointer to Structure (required)\n"
"    energy_cutoff  hbond energy cutoff (default -0.5)\n"
"    min_helix_len  minimum helix length (default 3)\n"
"    min_strand_len minimum strand length (default 3)\n"
"    report         whether to log computed values (default false)\n"
"    return_values  whether to return computed values (default false)\n";

static PyMethodDef dssp_methods[] = {
    { PY_STUPID "compute_ss", (PyCFunction) compute_ss, METH_VARARGS|METH_KEYWORDS, PY_STUPID docstr_compute_ss },
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
