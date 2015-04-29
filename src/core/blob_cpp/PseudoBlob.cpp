// vi: set expandtab ts=4 sw=4:

#include <stddef.h>

#include "AtomBlob.h"
#include "atomstruct/AtomicStructure.h"
#include "PseudoBlob.h"
#include "numpy_common.h"
#include "blob_op.h"
#include "set_blob.h"

namespace blob {
    
template PyObject* new_blob<PseudoBlob>(PyTypeObject*);

extern "C" {
    
static void
PseudoBlob_dealloc(PyObject* obj)
{
    PseudoBlob* self = static_cast<PseudoBlob*>(obj);
    delete self->_items;
    if (self->_weaklist)
        PyObject_ClearWeakRefs(obj);
    obj->ob_type->tp_free(obj);
    
}

static const char PseudoBlob_doc[] = "PseudoBlob documentation";

static PyObject*
pb_atoms(PyObject* self, void*)
{
    PseudoBlob* pb = static_cast<PseudoBlob*>(self);
    PyObject* py_ab1 = new_blob<AtomBlob>(&AtomBlob_type);
    AtomBlob* ab1 = static_cast<AtomBlob*>(py_ab1);
    PyObject* py_ab2 = new_blob<AtomBlob>(&AtomBlob_type);
    AtomBlob* ab2 = static_cast<AtomBlob*>(py_ab2);
    for (auto& b: *pb->_items) {
        auto& atoms = b->atoms();
        ab1->_items->emplace_back(atoms[0]);
        ab2->_items->emplace_back(atoms[1]);
    }
    PyObject* blob_list = PyTuple_New(2);
    if (blob_list == NULL)
        return NULL;
    PyTuple_SET_ITEM(blob_list, 0, py_ab1);
    PyTuple_SET_ITEM(blob_list, 1, py_ab2);
    return blob_list;
}

static PyObject*
pb_colors(PyObject* self, void*)
{
    PseudoBlob* pb = static_cast<PseudoBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[2] = {(unsigned int)pb->_items->size(), 4};
    PyObject* colors = allocate_python_array(2, shape, NPY_UINT8);
    unsigned char* color_data = (unsigned char*) PyArray_DATA((
        PyArrayObject*)colors);
    for (auto b: *(pb->_items)) {
        auto rgba = b->color();
        *color_data++ = rgba.r;
        *color_data++ = rgba.g;
        *color_data++ = rgba.b;
        *color_data++ = rgba.a;
    }
    return colors;
}

static int
pb_set_colors(PyObject* self, PyObject* value, void*)
{
    return set_blob<PseudoBlob>(self, value, &atomstruct::PBond::set_color);
}

static PyObject*
pb_displays(PyObject* self, void*)
{
    PseudoBlob* pb = static_cast<PseudoBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)pb->_items->size()};
    PyObject* displays = allocate_python_array(1, shape, NPY_UINT8);
    unsigned char* data = (unsigned char*) PyArray_DATA((PyArrayObject*)displays);
    for (auto b: *(pb->_items)) {
        *data++ = static_cast<unsigned char>(b->display());
    }
    return displays;
}

static int
pb_set_displays(PyObject* self, PyObject* value, void*)
{
    return set_blob<PseudoBlob>(self, value, &atomstruct::PBond::set_display);
}

static PyObject*
pb_halfbonds(PyObject* self, void*)
{
    PseudoBlob* pb = static_cast<PseudoBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)pb->_items->size()};
    PyObject* halfbonds = allocate_python_array(1, shape, NPY_BOOL);
    unsigned char* data = (unsigned char*) PyArray_DATA((PyArrayObject*)halfbonds);
    for (auto b: *(pb->_items)) {
        *data++ = b->halfbond();
    }
    return halfbonds;
}

static int
pb_set_halfbonds(PyObject* self, PyObject* value, void*)
{
    return set_blob<PseudoBlob>(self, value, &atomstruct::PBond::set_halfbond);
}

static PyObject*
pb_radii(PyObject* self, void*)
{
    PseudoBlob* pb = static_cast<PseudoBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)pb->_items->size()};
    PyObject* radii = allocate_python_array(1, shape, NPY_FLOAT);
    float* data = (float*) PyArray_DATA((PyArrayObject*)radii);
    for (auto b: *(pb->_items)) {
        *data++ = b->radius();
    }
    return radii;
}

static int
pb_set_radii(PyObject* self, PyObject* value, void*)
{
    return set_blob<PseudoBlob>(self, value, &atomstruct::PBond::set_radius);
}

static PyMethodDef PseudoBlob_methods[] = {
    { (char*)"filter", blob_filter<PseudoBlob>, METH_O,
        (char*)"filter pseudobond blob based on array/list of booleans" },
    { (char*)"intersect", blob_intersect<PseudoBlob>, METH_O,
        (char*)"intersect pseudobond blobs" },
    { (char*)"merge", blob_merge<PseudoBlob>, METH_O,
        (char*)"merge atom blobs" },
    { (char*)"subtract", blob_subtract<PseudoBlob>, METH_O,
        (char*)"subtract atom blobs" },
    { NULL, NULL, 0, NULL }
};

static PyNumberMethods PseudoBlob_as_number = {
    0,                                      // nb_add
    (binaryfunc)blob_subtract<PseudoBlob>,  // nb_subtract
    0,                                      // nb_multiply
    0,                                      // nb_remainder
    0,                                      // nb_divmod
    0,                                      // nb_power
    0,                                      // nb_negative
    0,                                      // nb_positive
    0,                                      // nb_absolute
    0,                                      // nb_bool
    0,                                      // nb_invert
    0,                                      // nb_lshift
    0,                                      // nb_rshift
    (binaryfunc)blob_intersect<PseudoBlob>, // nb_and
    0,                                      // nb_xor
    (binaryfunc)blob_merge<PseudoBlob>,     // nb_or
    0,                                      // nb_int
    0,                                      // nb_reserved
    0,                                      // nb_float
    0,                                      // nb_inplace_add
    0,                                      // nb_inplace_subtract
    0,                                      // nb_inplace_multiply
    0,                                      // nb_inplace_remainder
    0,                                      // nb_inplace_power
    0,                                      // nb_inplace_lshift
    0,                                      // nb_inplace_rshift
    0,                                      // nb_inplace_and
    0,                                      // nb_inplace_xor
    0,                                      // nb_inplace_or
};

static PyGetSetDef PseudoBlob_getset[] = {
    { (char*)"atoms", pb_atoms, NULL,
        (char*)"2-tuple of atom blobs", NULL},
    { (char*)"colors", pb_colors, pb_set_colors,
        (char*)"numpy Nx4 array of (unsigned char) RGBA values", NULL},
    { (char*)"displays", pb_displays, pb_set_displays,
        (char*)"numpy array of display values (0/1/2 = never/smart/always)", NULL},
    { (char*)"halfbonds", pb_halfbonds, pb_set_halfbonds,
        (char*)"numpy array of (bool) halfbonds", NULL},
    { (char*)"radii", pb_radii, pb_set_radii,
        (char*)"numpy array of (float) radii", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

static PyMappingMethods PseudoBlob_len = { blob_len<PseudoBlob>, NULL, NULL };

} // extern "C"

PyTypeObject PseudoBlob_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "structaccess.PseudoBlob", // tp_name
    sizeof (PseudoBlob), // tp_basicsize
    0, // tp_itemsize
    PseudoBlob_dealloc, // tp_dealloc
    0, // tp_print
    0, // tp_getattr
    0, // tp_setattr
    0, // tp_reserved
    0, // tp_repr
    &PseudoBlob_as_number, // tp_as_number
    0, // tp_as_sequence
    &PseudoBlob_len, // tp_as_mapping
    0, // tp_hash
    0, // tp_call
    0, // tp_str
    0, // tp_getattro
    0, // tp_setattro
    0, // tp_as_buffer
    Py_TPFLAGS_DEFAULT, // tp_flags
    PseudoBlob_doc, // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    offsetof(PseudoBlob, _weaklist), // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    PseudoBlob_methods, // tp_methods
    0, // tp_members
    PseudoBlob_getset, // tp_getset
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
