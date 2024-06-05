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

#include <Python.h>
#include <algorithm>    // std::min
#include <math.h>
#include <memory>
#include <string>
#include <vector>

#include <atomstruct/Chain.h>
#include <atomstruct/search.h>
#include <logger/logger.h>

using atomstruct::Atom;
using atomstruct::Chain;
using atomstruct::AtomSearchTree;

class EndPoint
{
public:
    Chain*  chain;
    Chain::SeqPos  pos;

    EndPoint(Chain* _chain, Chain::SeqPos _pos): chain(_chain), pos(_pos) {};
};

class Link
{
public:
    std::vector<EndPoint*> info;
    float val;
    float penalty = 0.0;
    std::vector<std::shared_ptr<Link>> cross_links;

    Link(EndPoint* e1, EndPoint* e2, float _val) {
        info.push_back(e1);
        info.push_back(e2);
        val = _val;
    }
    ~Link() {
        for (auto ep: info)
            delete ep;
    }
};

PyObject *
multi_align(std::vector<const Chain*>& chains, double dist_cutoff, bool col_all, char gap_char,
    bool circular, const char* status_prefix, PyObject* py_logger)
{
    PyErr_SetString(PyExc_NotImplementedError, "C++ multi_align not implemented");
    return nullptr;
    //TODO: once enough of the below implemented, remove the above two lines

    // Create list of parings between chains and prune to be monotonic
    std::map<const Chain, AtomSearchTree> trees;

    // For each pair, go through the second chain residue by residue
    // and compile crosslinks to other chain.  As links are compiled,
    // figure out what previous links are crossed and keep a running
    // "penalty" for links based on what they cross.  Sort links by
    // penalty and keep pruning worst link until no links cross.
    std::vector<Link*> all_links;

    std::map<const Chain*, std::vector<Atom*>> pas;
    std::vector<std::vector<std::shared_ptr<Link>>> pairings;
    logger::status(py_logger, status_prefix, "Finding residue principal atoms");
    //TODO: lots
}

static PyObject*
py_multi_align(PyObject*, PyObject* args)
{
    PyObject* chain_ptrs_list;
    PyObject* py_logger;
    double dist_cutoff;
    int col_all, circular, py_gap_char;
    const char* status_prefix;
    if (!PyArg_ParseTuple(args, const_cast<char *>("OfpCpsO"),
            &chain_ptrs_list, &dist_cutoff, &col_all, &py_gap_char, &circular, &status_prefix, &py_logger))
        return NULL;
    char gap_char = (char)py_gap_char;
    if (!PySequence_Check(chain_ptrs_list)) {
        PyErr_SetString(PyExc_TypeError, "First arg is not a sequence of Chain pointers");
        return nullptr;
    }
    auto num_chains = PySequence_Size(chain_ptrs_list);
    if (num_chains < 2) {
        PyErr_SetString(PyExc_ValueError, "First arg (sequence of chain pointers) must contain at least"
            " two chains");
        return nullptr;
    }
    std::vector<const Chain*> chains;
    for (decltype(num_chains) i = 0; i < num_chains; ++i) {
        PyObject* py_ptr = PySequence_GetItem(chain_ptrs_list, i);
        if (!PyLong_Check(py_ptr)) {
            std::stringstream err_msg;
            err_msg << "Item at index " << i << " of first arg is not an int (chain pointer)";
            PyErr_SetString(PyExc_TypeError, err_msg.str().c_str());
            return nullptr;
        }
        chains.push_back(static_cast<const Chain*>(PyLong_AsVoidPtr(py_ptr)));
    }
    return multi_align(chains, dist_cutoff, (bool)col_all, gap_char, (bool)circular,
        status_prefix, py_logger);
}

static struct PyMethodDef msa3d_methods[] =
{
  {const_cast<char*>("multi_align"), py_multi_align, METH_VARARGS, NULL},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef msa3d_def = {
        PyModuleDef_HEAD_INIT,
        "_msa3d",
        "Compute alignment from 3D superposition",
        -1,
        msa3d_methods,
        nullptr,
        nullptr,
        nullptr,
        nullptr
};

// ----------------------------------------------------------------------------
// Initialization routine called by python when module is dynamically loaded.
//
PyMODINIT_FUNC
PyInit__msa3d()
{
    return PyModule_Create(&msa3d_def);
}
