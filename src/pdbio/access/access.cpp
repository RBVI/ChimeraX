// vim: set expandtab ts=4 sw=4:
#include "blob/MolBlob.h"
#include "blob/ResBlob.h"
#include "blob/AtomBlob.h"
#include "molecule/Molecule.h"
#include "molecule/Residue.h"
#include "molecule/Atom.h"
#include "molecule/Bond.h"
#include "base-geom/Coord.h"
#include "molecule/Element.h"
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>  // std::ostringstream

extern "C" {

static struct PyMethodDef access_functions[] =
{
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef access_module =
{
    PyModuleDef_HEAD_INIT,
    "access",
    "Access functions for molecular aggregates",
    -1,
    access_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_access()
{
    MolBlob_type.tp_new = PyType_GenericNew;
    ResBlob_type.tp_new = PyType_GenericNew;
    AtomBlob_type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MolBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&ResBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&AtomBlob_type) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&access_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MolBlob_type);
    Py_INCREF(&ResBlob_type);
    Py_INCREF(&AtomBlob_type);
    // make blob types visible so their doc strings can be accessed
    PyModule_AddObject(m, "MolBlob", (PyObject *)&MolBlob_type);
    PyModule_AddObject(m, "ResBlob", (PyObject *)&ResBlob_type);
    PyModule_AddObject(m, "AtomBlob", (PyObject *)&AtomBlob_type);
    return m;
}

}  // extern "C"
