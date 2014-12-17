// vim: set expandtab ts=4 sw=4:
#include "Blob.h"
#include "AtomBlob.h"
#include "ResBlob.h"
#include "numpy_common.h"
#include <stddef.h>

namespace blob {
    
template PyObject* newBlob<AtomBlob>(PyTypeObject*);

extern "C" {
    
static void
AtomBlob_dealloc(PyObject* obj)
{
    AtomBlob* self = static_cast<AtomBlob*>(obj);
    delete self->_items;
    if (self->_weaklist)
        PyObject_ClearWeakRefs(obj);
    obj->ob_type->tp_free(obj);
    
}

static const char AtomBlob_doc[] = "AtomBlob documentation";

static PyObject*
ab_coords(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    initialize_numpy();
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[2] = {(unsigned int)ab->_items->size(), 3};
    PyObject* coords = allocate_python_array(2, shape, NPY_DOUBLE);
    double* crd_data = (double*) PyArray_DATA((PyArrayObject*)coords);
    auto& atoms = ab->_items;
    for (auto ai = atoms->begin(); ai != atoms->end(); ++ai) {
        auto& crd = (*ai)->coord();
        *crd_data++ = crd[0];
        *crd_data++ = crd[1];
        *crd_data++ = crd[2];
    }
    return coords;
}

static PyObject*
ab_element_names(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    PyObject *list = PyList_New(ab->_items->size());
    int i = 0; // auto apparently can't be used if the types differ,
                // so can't easily declare this in the loop
    for (auto ai = ab->_items->begin(); ai != ab->_items->end(); ++ai, ++i) {
        PyList_SetItem(list, i, PyUnicode_FromString((*ai)->element().name()));
    }
    return list;
}

static PyObject*
ab_element_numbers(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    initialize_numpy();
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)ab->_items->size()};
    PyObject* element_numbers = allocate_python_array(1, shape, NPY_UBYTE);
    unsigned char* data = (unsigned char*) PyArray_DATA(
        (PyArrayObject*)element_numbers);
    auto& atoms = ab->_items;
    for (auto ai = atoms->begin(); ai != atoms->end(); ++ai) {
        *data++ = (*ai)->element().number();
    }
    return element_numbers;
}

static PyObject*
ab_names(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    PyObject *list = PyList_New(ab->_items->size());
    int i = 0;
    for (auto ai = ab->_items->begin(); ai != ab->_items->end(); ++ai, ++i){
        PyList_SetItem(list, i, PyUnicode_FromString((*ai)->name().c_str()));
    }
    return list;
}

static PyObject*
ab_residues(PyObject* self, void*)
{
    PyObject* py_rb = newBlob<ResBlob>(&ResBlob_type);
    ResBlob* rb = static_cast<ResBlob*>(py_rb);
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    for (auto ai = ab->_items->begin(); ai != ab->_items->end(); ++ai) {
        rb->_items->emplace_back((*ai)->residue());
    }
    return py_rb;
}

static PyMethodDef AtomBlob_methods[] = {
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef AtomBlob_getset[] = {
    { (char*)"coords", ab_coords, NULL,
        (char*)"numpy Nx3 array of atom coordinates", NULL},
    { (char*)"element_names", ab_element_names, NULL,
        (char*)"list of element names", NULL},
    { (char*)"element_numbers", ab_element_numbers, NULL,
        (char*)"numpy array of element numbers", NULL},
    { (char*)"names", ab_names, NULL, (char*)"list of atom names", NULL},
    { (char*)"residues", ab_residues, NULL, (char*)"ResBlob", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

} // extern "C"

PyTypeObject AtomBlob_type = {
    PyObject_HEAD_INIT(NULL)
    "structaccess.AtomBlob", // tp_name
    sizeof (AtomBlob), // tp_basicsize
    0, // tp_itemsize
    AtomBlob_dealloc, // tp_dealloc
    0, // tp_print
    0, // tp_getattr
    0, // tp_setattr
    0, // tp_reserved
    0, // tp_repr
    0, // tp_as_number
    0, // tp_as_sequence
    0, // tp_as_mapping
    0, // tp_hash
    0, // tp_call
    0, // tp_str
    0, // tp_getattro
    0, // tp_setattro
    0, // tp_as_buffer
    Py_TPFLAGS_DEFAULT, // tp_flags
    AtomBlob_doc, // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    offsetof(Blob, _weaklist), // tp_weaklist
    0, // tp_iter
    0, // tp_iternext
    AtomBlob_methods, // tp_methods
    0, // tp_members
    AtomBlob_getset, // tp_getset
    0, // tp_base
    0, // tp_dict
    0, // tp_descr_get
    0, // tp_descr_set
    0, // tp_dict_offset
    0, // tp_init,
    0, // tp_alloc
    0, // tp_new
    0, // tp_free
    0, // tp_is_gc
    0, // tp_bases
    0, // tp_mro
    0, // tp_cache
    0, // tp_subclasses
    0, // tp_weaklist
};

}  // namespace blob
