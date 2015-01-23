// vi: set expandtab ts=4 sw=4:
#include "Blob.h"
#include "AtomBlob.h"
#include "ResBlob.h"
#include "numpy_common.h"
#include "set_blob.h"
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
ab_colors(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[2] = {(unsigned int)ab->_items->size(), 4};
    PyObject* colors = allocate_python_array(2, shape, NPY_UINT8);
    unsigned char* color_data = (unsigned char*) PyArray_DATA((
        PyArrayObject*)colors);
    for (auto a: *(ab->_items)) {
        auto rgba = a->color();
        *color_data++ = rgba.r;
        *color_data++ = rgba.g;
        *color_data++ = rgba.b;
        *color_data++ = rgba.a;
    }
    return colors;
}

static int
ab_set_colors(PyObject* self, PyObject* value, void*)
{
    return set_blob<AtomBlob>(self, value, &atomstruct::Atom::set_color);
}

static PyObject*
ab_coords(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[2] = {(unsigned int)ab->_items->size(), 3};
    PyObject* coords = allocate_python_array(2, shape, NPY_DOUBLE);
    double* crd_data = (double*) PyArray_DATA((PyArrayObject*)coords);
    for (auto a: *(ab->_items)) {
        auto& crd = a->coord();
        *crd_data++ = crd[0];
        *crd_data++ = crd[1];
        *crd_data++ = crd[2];
    }
    return coords;
}

static PyObject*
ab_displays(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)ab->_items->size()};
    PyObject* displays = allocate_python_array(1, shape, NPY_BOOL);
    unsigned char* data = (unsigned char*) PyArray_DATA((PyArrayObject*)displays);
    for (auto a: *(ab->_items)) {
        *data++ = a->display();
    }
    return displays;
}

static int
ab_set_displays(PyObject* self, PyObject* value, void*)
{
    return set_blob<AtomBlob>(self, value, &atomstruct::Atom::set_display);
}

static PyObject*
ab_draw_modes(PyObject* self, void*)
{
    AtomBlob* ab = static_cast<AtomBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)ab->_items->size()};
    PyObject* draw_modes = allocate_python_array(1, shape, NPY_INT8);
    signed char* data = (signed char*) PyArray_DATA((PyArrayObject*)draw_modes);
    for (auto a: *(ab->_items)) {
        *data++ = a->draw_mode();
    }
    return draw_modes;
}

static int
ab_set_draw_modes(PyObject* self, PyObject* value, void*)
{
    return set_blob<AtomBlob>(self, value, &atomstruct::Atom::set_draw_mode);
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
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)ab->_items->size()};
    PyObject* element_numbers = allocate_python_array(1, shape, NPY_UBYTE);
    unsigned char* data = (unsigned char*) PyArray_DATA(
        (PyArrayObject*)element_numbers);
    for (auto a: *(ab->_items)) {
        *data++ = a->element().number();
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
    for (auto a: *(ab->_items)) {
        rb->_items->emplace_back(a->residue());
    }
    return py_rb;
}

static PyMethodDef AtomBlob_methods[] = {
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef AtomBlob_getset[] = {
    { (char*)"colors", ab_colors, ab_set_colors,
        (char*)"numpy Nx4 array of (unsigned char) RGBA values", NULL},
    { (char*)"coords", ab_coords, NULL,
        (char*)"numpy Nx3 array of atom coordinates", NULL},
    { (char*)"displays", ab_displays, ab_set_displays,
        (char*)"numpy array of (bool) displays", NULL},
    { (char*)"draw_modes", ab_draw_modes, ab_set_draw_modes,
        (char*)"numpy array of (int) draw modes", NULL},
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
    PyVarObject_HEAD_INIT(NULL, 0)
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
    offsetof(AtomBlob, _weaklist), // tp_weaklistoffset
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
