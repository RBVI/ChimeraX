// vi: set expandtab ts=4 sw=4:
#include "Blob.h"
#include "ResBlob.h"
#include <atomstruct/Residue.h>
#include "numpy_common.h"
#include "blob_op.h"
#include <stddef.h>

namespace blob {

template PyObject* new_blob<ResBlob>(PyTypeObject*);

extern "C" {
    
static void
ResBlob_dealloc(PyObject* obj)
{
    ResBlob* self = static_cast<ResBlob*>(obj);
    delete self->_items;
    if (self->_weaklist)
        PyObject_ClearWeakRefs(obj);
    obj->ob_type->tp_free(obj);
    
}

static const char ResBlob_doc[] = "ResBlob documentation";

static PyObject*
rb_chain_ids(PyObject* self, void*)
{
    ResBlob* rb = static_cast<ResBlob*>(self);
    PyObject *list = PyList_New(rb->_items->size());
    int i = 0;
    for (auto ri = rb->_items->begin(); ri != rb->_items->end(); ++ri, ++i){
        PyList_SetItem(list, i,
            PyUnicode_FromString((*ri)->chain_id().c_str()));
    }
    return list;
}

static PyObject*
rb_names(PyObject* self, void*)
{
    ResBlob* rb = static_cast<ResBlob*>(self);
    PyObject *list = PyList_New(rb->_items->size());
    int i = 0;
    for (auto ri = rb->_items->begin(); ri != rb->_items->end(); ++ri, ++i){
        PyList_SetItem(list, i, PyUnicode_FromString((*ri)->name().c_str()));
    }
    return list;
}

static PyObject*
rb_numbers(PyObject* self, void*)
{
    ResBlob* rb = static_cast<ResBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)rb->_items->size()};
    PyObject* residue_numbers = allocate_python_array(1, shape, NPY_INT);
    int* data = (int*) PyArray_DATA((PyArrayObject*)residue_numbers);
    for (auto ri = rb->_items->begin(); ri != rb->_items->end(); ++ri){
        *data++ = (*ri)->position();
    }
    return residue_numbers;
}

static PyObject*
rb_strs(PyObject* self, void*)
{
    ResBlob* rb = static_cast<ResBlob*>(self);
    PyObject *list = PyList_New(rb->_items->size());
    int i = 0;
    for (auto ri = rb->_items->begin(); ri != rb->_items->end(); ++ri, ++i){
        PyList_SetItem(list, i, PyUnicode_FromString((*ri)->str().c_str()));
    }
    return list;
}

static PyMethodDef ResBlob_methods[] = {
    { (char*)"filter", blob_filter<ResBlob>, METH_O,
        (char*)"filter residue blob based on array/list of booleans" },
    { (char*)"merge", blob_merge<ResBlob>, METH_O,
        (char*)"merge residue blobs" },
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef ResBlob_getset[] = {
    { "chain_ids", rb_chain_ids, NULL, "list of chain IDs", NULL},
    { "names", rb_names, NULL, "list of residue names", NULL},
    { "numbers", rb_numbers, NULL,
        "numpy array of residue sequence numbers", NULL},
    { "strs", rb_strs, NULL,
        "list of human-friendly residue identifiers", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

} // extern "C"

PyTypeObject ResBlob_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "structaccess.ResBlob", // tp_name
    sizeof (ResBlob), // tp_basicsize
    0, // tp_itemsize
    ResBlob_dealloc, // tp_dealloc
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
    ResBlob_doc, // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    offsetof(ResBlob, _weaklist), // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    ResBlob_methods, // tp_methods
    0, // tp_members
    ResBlob_getset, // tp_getset
    0, // tp_base
    0, // tp_dict
    0, // tp_descr_get
    0, // tp_descr_set
    0, // tp_dict_offset
    0, // tp_init,
    PyType_GenericAlloc, // tp_alloc
    0, // tp_new
    PyObject_Free, // tp_free
    0, // tp_is_gc
    0, // tp_bases
    0, // tp_mro
    0, // tp_cache
    0, // tp_subclasses
    0, // tp_weaklist
};

}  // namespace blob
