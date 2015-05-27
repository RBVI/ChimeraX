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
    delete self->_observer;
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

static PyObject*
rb_unique_ids(PyObject* self, void*)
{
    ResBlob* rb = static_cast<ResBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)rb->_items->size()};
    PyObject* unique_ids = allocate_python_array(1, shape, NPY_INT);
    int* data = (int*) PyArray_DATA((PyArrayObject*)unique_ids);
    // TODO: Don't assume residue atoms are consecutive.
    int rid = -1;
    const atomstruct::Residue *rprev = NULL;
    for (auto ri = rb->_items->begin(); ri != rb->_items->end(); ++ri){
        const atomstruct::Residue *r = ri->get();
	if (rprev == NULL || r->position() != rprev->position() || r->chain_id() != rprev->chain_id()) {
	    rid += 1;
	    rprev = r;
	}
        *data++ = rid;
    }
    return unique_ids;
}

static PyMethodDef ResBlob_methods[] = {
    { (char*)"filter", blob_filter<ResBlob>, METH_O,
        (char*)"filter residue blob based on array/list of booleans" },
    { (char*)"intersect", blob_intersect<ResBlob>, METH_O,
        (char*)"intersect residue blobs" },
    { (char*)"merge", blob_merge<ResBlob>, METH_O,
        (char*)"merge atom blobs" },
    { (char*)"subtract", blob_subtract<ResBlob>, METH_O,
        (char*)"subtract atom blobs" },
    { NULL, NULL, 0, NULL }
};

static PyNumberMethods ResBlob_as_number = {
    0,                                   // nb_add
    (binaryfunc)blob_subtract<ResBlob>,  // nb_subtract
    0,                                   // nb_multiply
    0,                                   // nb_remainder
    0,                                   // nb_divmod
    0,                                   // nb_power
    0,                                   // nb_negative
    0,                                   // nb_positive
    0,                                   // nb_absolute
    0,                                   // nb_bool
    0,                                   // nb_invert
    0,                                   // nb_lshift
    0,                                   // nb_rshift
    (binaryfunc)blob_intersect<ResBlob>, // nb_and
    0,                                   // nb_xor
    (binaryfunc)blob_merge<ResBlob>,     // nb_or
    0,                                   // nb_int
    0,                                   // nb_reserved
    0,                                   // nb_float
    0,                                   // nb_inplace_add
    0,                                   // nb_inplace_subtract
    0,                                   // nb_inplace_multiply
    0,                                   // nb_inplace_remainder
    0,                                   // nb_inplace_power
    0,                                   // nb_inplace_lshift
    0,                                   // nb_inplace_rshift
    0,                                   // nb_inplace_and
    0,                                   // nb_inplace_xor
    0,                                   // nb_inplace_or
};

static PyGetSetDef ResBlob_getset[] = {
    { (char*)"chain_ids", rb_chain_ids, NULL,
        (char*)"list of chain IDs", NULL},
    { (char*)"names", rb_names, NULL,
        (char*)"list of residue names", NULL},
    { (char*)"numbers", rb_numbers, NULL,
        (char*)"numpy array of residue sequence numbers", NULL},
    { (char*)"strs", rb_strs, NULL,
        (char*)"list of human-friendly residue identifiers", NULL},
    { (char*)"unique_ids", rb_unique_ids, NULL,
        (char*)"numpy array of integer ids unique for each chain and residue number", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

static PyMappingMethods ResBlob_len = { blob_len<ResBlob>, NULL, NULL };

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
    &ResBlob_as_number, // tp_as_number
    0, // tp_as_sequence
    &ResBlob_len, // tp_as_mapping
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
