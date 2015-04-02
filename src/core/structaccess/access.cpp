// vi: set expandtab ts=4 sw=4:
#include <blob/AtomBlob.h>
#include <blob/BondBlob.h>
#include <blob/PseudoBlob.h>
#include <blob/ResBlob.h>
#include <blob/StructBlob.h>

extern "C" {

static struct PyMethodDef structaccess_functions[] =
{
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef structaccess_module =
{
    PyModuleDef_HEAD_INIT,
    "structaccess",
    "Access functions for molecular aggregates",
    -1,
    structaccess_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_structaccess()
{
    using blob::StructBlob_type;
    using blob::ResBlob_type;
    using blob::AtomBlob_type;
    using blob::BondBlob_type;
    using blob::PseudoBlob_type;
    StructBlob_type.tp_new = blob::PyType_NewBlob<blob::StructBlob>;
    ResBlob_type.tp_new = blob::PyType_NewBlob<blob::ResBlob>;
    AtomBlob_type.tp_new = blob::PyType_NewBlob<blob::AtomBlob>;
    BondBlob_type.tp_new = blob::PyType_NewBlob<blob::BondBlob>;
    PseudoBlob_type.tp_new = blob::PyType_NewBlob<blob::PseudoBlob>;
    if (PyType_Ready(&StructBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&ResBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&AtomBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&BondBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&PseudoBlob_type) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&structaccess_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&StructBlob_type);
    Py_INCREF(&ResBlob_type);
    Py_INCREF(&AtomBlob_type);
    Py_INCREF(&BondBlob_type);
    Py_INCREF(&PseudoBlob_type);
    // make blob types visible so their doc strings can be accessed
    PyModule_AddObject(m, "StructBlob", (PyObject *)&StructBlob_type);
    PyModule_AddObject(m, "ResBlob", (PyObject *)&ResBlob_type);
    PyModule_AddObject(m, "AtomBlob", (PyObject *)&AtomBlob_type);
    PyModule_AddObject(m, "BondBlob", (PyObject *)&BondBlob_type);
    PyModule_AddObject(m, "PseudoBlob", (PyObject *)&PseudoBlob_type);
    return m;
}

}  // extern "C"
