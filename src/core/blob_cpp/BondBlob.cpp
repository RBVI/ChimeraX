// vi: set expandtab ts=4 sw=4:

#include <stddef.h>

#include "BondBlob.h"
#include "numpy_common.h"
#include "set_blob.h"

namespace blob {
    
template PyObject* newBlob<BondBlob>(PyTypeObject*);

extern "C" {
    
static void
BondBlob_dealloc(PyObject* obj)
{
    BondBlob* self = static_cast<BondBlob*>(obj);
    delete self->_items;
    if (self->_weaklist)
        PyObject_ClearWeakRefs(obj);
    obj->ob_type->tp_free(obj);
    
}

static const char BondBlob_doc[] = "BondBlob documentation";

static PyObject*
bb_colors(PyObject* self, void*)
{
    BondBlob* bb = static_cast<BondBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[2] = {(unsigned int)bb->_items->size(), 4};
    PyObject* colors = allocate_python_array(2, shape, NPY_UINT8);
    unsigned char* color_data = (unsigned char*) PyArray_DATA((
        PyArrayObject*)colors);
    for (auto b: *(bb->_items)) {
        auto rgba = b->color();
        *color_data++ = rgba.r;
        *color_data++ = rgba.g;
        *color_data++ = rgba.b;
        *color_data++ = rgba.a;
    }
    return colors;
}

static int
bb_set_colors(PyObject* self, PyObject* value, void*)
{
    return set_blob<BondBlob>(self, value, &atomstruct::Bond::set_color);
}

static PyObject*
bb_displays(PyObject* self, void*)
{
    BondBlob* bb = static_cast<BondBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)bb->_items->size()};
    PyObject* displays = allocate_python_array(1, shape, NPY_BOOL);
    unsigned char* data = (unsigned char*) PyArray_DATA((PyArrayObject*)displays);
    for (auto b: *(bb->_items)) {
        *data++ = b->display();
    }
    return displays;
}

static int
bb_set_displays(PyObject* self, PyObject* value, void*)
{
    return set_blob<BondBlob>(self, value, &atomstruct::Bond::set_display);
}

static PyObject*
bb_halfbonds(PyObject* self, void*)
{
    BondBlob* bb = static_cast<BondBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)bb->_items->size()};
    PyObject* halfbonds = allocate_python_array(1, shape, NPY_BOOL);
    unsigned char* data = (unsigned char*) PyArray_DATA((PyArrayObject*)halfbonds);
    for (auto b: *(bb->_items)) {
        *data++ = b->halfbond();
    }
    return halfbonds;
}

static int
bb_set_halfbonds(PyObject* self, PyObject* value, void*)
{
    return set_blob<BondBlob>(self, value, &atomstruct::Bond::set_halfbond);
}

static PyObject*
bb_radii(PyObject* self, void*)
{
    BondBlob* bb = static_cast<BondBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)bb->_items->size()};
    PyObject* radii = allocate_python_array(1, shape, NPY_FLOAT);
    float* data = (float*) PyArray_DATA((PyArrayObject*)radii);
    for (auto b: *(bb->_items)) {
        *data++ = b->radius();
    }
    return radii;
}

static int
bb_set_radii(PyObject* self, PyObject* value, void*)
{
    return set_blob<BondBlob>(self, value, &atomstruct::Bond::set_radius);
}

static PyMethodDef BondBlob_methods[] = {
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef BondBlob_getset[] = {
    { (char*)"colors", bb_colors, bb_set_colors,
        (char*)"numpy Nx4 array of (unsigned char) RGBA values", NULL},
    { (char*)"displays", bb_displays, bb_set_displays,
        (char*)"numpy array of (bool) displays", NULL},
    { (char*)"halfbonds", bb_halfbonds, bb_set_halfbonds,
        (char*)"numpy array of (bool) halfbonds", NULL},
    { (char*)"radii", bb_radii, bb_set_radii,
        (char*)"numpy array of (float) radii", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

} // extern "C"

PyTypeObject BondBlob_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "structaccess.BondBlob", // tp_name
    sizeof (BondBlob), // tp_basicsize
    0, // tp_itemsize
    BondBlob_dealloc, // tp_dealloc
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
    BondBlob_doc, // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    offsetof(BondBlob, _weaklist), // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    BondBlob_methods, // tp_methods
    0, // tp_members
    BondBlob_getset, // tp_getset
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
