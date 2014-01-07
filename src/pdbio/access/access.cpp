// vim: set expandtab ts=4 sw=4:
#include "access.h"
#include "MolBlob.h"
#include "ResBlob.h"
#include "AtomBlob.h"
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

// Need to call NumPy import_array() before using NumPy routines
void *
initialize_numpy()
{
    static bool first_call = true;
    if (first_call) {
        first_call = false;
        import_array();
    }
    return NULL;
}

static const char *
numpy_type_name(int type)
{
    const char *name = "unknown";
    switch (type) {
        case NPY_BOOL: name = "bool"; break;
        case NPY_BYTE: name = "byte"; break;
        case NPY_UBYTE: name = "ubyte"; break;
        case NPY_SHORT: name = "short"; break;
        case NPY_USHORT: name = "ushort"; break;
        case NPY_INT: name = "int"; break;
        case NPY_UINT: name = "uint"; break;
        case NPY_LONG: name = "long"; break;
        case NPY_ULONG: name = "ulong"; break;
        case NPY_LONGLONG: name = "longlong"; break;
        case NPY_ULONGLONG: name = "ulonglong"; break;
        case NPY_FLOAT: name = "float"; break;
        case NPY_DOUBLE: name = "double"; break;
        case NPY_LONGDOUBLE: name = "longdouble"; break;
        case NPY_CFLOAT: name = "cfloat"; break;
        case NPY_CDOUBLE: name = "cdouble"; break;
        case NPY_CLONGDOUBLE: name = "clongdouble"; break;
        case NPY_OBJECT: name = "object"; break;
        case NPY_STRING: name = "string"; break;
        case NPY_UNICODE: name = "unicode"; break;
        case NPY_VOID: name = "void"; break;
        default: break;
    }
    return name;
}

PyObject *
allocate_python_array(unsigned int dim, unsigned int *size, int type)
{
    npy_intp *sn = new npy_intp[dim];
    for (int i = 0; i < dim; ++i)
        sn[i] = (npy_intp)size[i];
    
    // array not initialized to zero
    PyObject *array = PyArray_SimpleNew(dim, sn, type);
    delete [] sn;
    if (array == NULL) {
        std::ostringstream msg;
        msg << numpy_type_name(type) << " array allocation of size (";
        for (int i = 0; i < dim; ++i) {
            msg << size[i] << (i < dim-1 ? ", " : "");
        }
        msg << ") failed " << std::endl;
        throw std::runtime_error(msg.str());
    }
    return array;
}

PyObject *
allocate_python_array(unsigned int dim, unsigned int *size, PyArray_Descr *dtype)
{
    npy_intp *sn = new npy_intp[dim];
    for (int i = 0; i < dim; ++i)
        sn[i] = (npy_intp)size[i];
    
    // array not initialized to zero
    PyObject *array = PyArray_SimpleNewFromDescr(dim, sn, dtype);
    delete [] sn;
    if (array == NULL) {
        std::ostringstream msg;
        msg << "Array allocation of size (";
        for (int i = 0; i < dim; ++i) {
            msg << size[i] << (i < dim-1 ? ", " : "");
        }
        msg << ") failed " << std::endl;
        throw std::runtime_error(msg.str());
    }
    return array;
}

// ifdef-ing out this no-longer-needed function to make it easier
// to resurrect if needed later
#if 0
static PyObject *
python_string_array(unsigned int size, int string_length, char **data)
{
    initialize_numpy();  // required before using NumPy

    PyArray_Descr *d = PyArray_DescrNewFromType(NPY_CHAR);
    d->elsize = string_length;
    unsigned int dimensions[1] = {size};
    PyObject *array = allocate_python_array(1, dimensions, d);
    if (data)
        *data = (char *)PyArray_DATA((PyArrayObject *)array);

    return array;
}
#endif

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
