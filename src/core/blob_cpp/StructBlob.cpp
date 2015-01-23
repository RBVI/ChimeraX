// vi: set expandtab ts=4 sw=4:
#include "StructBlob.h"
#include "AtomBlob.h"
#include "BondBlob.h"
#include "ResBlob.h"
#include <atomstruct/Bond.h>
#include "numpy_common.h"
#include "set_blob.h"
#include <unordered_map>
#include <stddef.h>

namespace blob {

using atomstruct::AtomicStructure;
using atomstruct::Atom;
using atomstruct::Bond;
    
template PyObject* newBlob<StructBlob>(PyTypeObject*);

extern "C" {
    
static void
StructBlob_dealloc(PyObject* obj)
{
    StructBlob* self = static_cast<StructBlob*>(obj);
    delete self->_items;
    if (self->_weaklist)
        PyObject_ClearWeakRefs(obj);
    obj->ob_type->tp_free(obj);
}

static const char StructBlob_doc[] = "StructBlob documentation";

static PyObject*
sb_atoms(PyObject* self, void*)
{
    PyObject* py_ab = newBlob<AtomBlob>(&AtomBlob_type);
    AtomBlob* ab = static_cast<AtomBlob*>(py_ab);
    StructBlob* sb = static_cast<StructBlob*>(self);
    for (auto& as: *sb->_items)
        for (auto& a: as->atoms()) 
            ab->_items->emplace_back(a.get());
    return py_ab;
}

static PyObject*
sb_bonds(PyObject* self, void*)
{
    PyObject* py_bb = newBlob<BondBlob>(&BondBlob_type);
    BondBlob* bb = static_cast<BondBlob*>(py_bb);
    StructBlob* sb = static_cast<StructBlob*>(self);
    for (auto& as: *sb->_items)
        for (auto& b: as->bonds()) 
            bb->_items->emplace_back(b.get());
    return py_bb;
}

static PyObject*
sb_bond_indices(PyObject* self, void*)
{
    StructBlob* sb = static_cast<StructBlob*>(self);
    PyObject* py_ab = sb_atoms(self, nullptr);
    AtomBlob* ab = static_cast<AtomBlob*>(py_ab);
    std::unordered_map<Atom *, AtomBlob::ItemsType::size_type> atom_map;
    decltype(atom_map)::mapped_type i = 0;
    auto& a_items = ab->_items;
    for (auto ai = a_items->begin(); ai != a_items->end(); ++ai, ++i) {
        atom_map[(*ai).get()] = i;
    }
    AtomicStructure::Bonds::size_type bonds_size = 0;
    auto& s_items = sb->_items;
    for (auto si = s_items->begin(); si != s_items->end(); ++si) {
        AtomicStructure *s = (*si).get();
        bonds_size += s->bonds().size();
    }
    // since the type of the shape array is unsigned int, don't bother
    // with sophisticated code to determine the size of indices, just
    // use NPY_INT, (not unsigned so that the Python code can do tricks
    // if it likes by using negative indices)
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    unsigned int shape[2] = {(unsigned int)bonds_size, 2};
    PyObject* bond_list = allocate_python_array(2, shape, NPY_INT);
    int* index_data = (int*) PyArray_DATA((PyArrayObject*)bond_list);
    for (auto& as: *s_items) {
        for (auto& b: as->bonds()) {
            *index_data++ = atom_map[b->atoms()[0]];
            *index_data++ = atom_map[b->atoms()[1]];
        }
    }
    return bond_list;
}

static PyObject*
sb_displays(PyObject* self, void*)
{
    StructBlob* sb = static_cast<StructBlob*>(self);
    if (PyArray_API == NULL)
        import_array1(NULL); // initialize NumPy
    static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
    unsigned int shape[1] = {(unsigned int)sb->_items->size()};
    PyObject* displays = allocate_python_array(1, shape, NPY_BOOL);
    unsigned char* data = (unsigned char*) PyArray_DATA((PyArrayObject*)displays);
    for (auto s: *(sb->_items)) {
        *data++ = s->display();
    }
    return displays;
}

static int
sb_set_displays(PyObject* self, PyObject* value, void*)
{
    return set_blob<StructBlob>(self, value,
        &atomstruct::AtomicStructure::set_display);
}

static PyObject*
sb_structures(PyObject* self, void*)
{
    StructBlob* sb = static_cast<StructBlob*>(self);
    PyObject *struct_list = PyList_New(sb->_items->size());
    StructBlob::ItemsType::size_type i = 0;
    for (auto si = sb->_items->begin(); si != sb->_items->end(); ++si) {
        PyObject* py_single_sb = newBlob<StructBlob>(&StructBlob_type);
        PyList_SET_ITEM(struct_list, i++, py_single_sb);
        StructBlob* single_sb = static_cast<StructBlob*>(py_single_sb);
        // since it's a shared_ptr, push_back, not emplace_back...
        single_sb->_items->push_back(*si);
    }
    return struct_list;
}

static PyObject*
sb_residues(PyObject* self, void*)
{
    PyObject* py_rb = newBlob<ResBlob>(&ResBlob_type);
    ResBlob* rb = static_cast<ResBlob*>(py_rb);
    StructBlob* sb = static_cast<StructBlob*>(self);
    for (auto si = sb->_items->begin(); si != sb->_items->end(); ++si) {
        const AtomicStructure::Residues& residues = (*si).get()->residues();
        for (auto ri = residues.begin(); ri != residues.end(); ++ri)
            rb->_items->emplace_back((*ri).get());
    }
    return py_rb;
}

static PyMethodDef StructBlob_methods[] = {
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef StructBlob_getset[] = {
    { "atoms", sb_atoms, NULL, "AtomBlob", NULL},
    { "bonds", sb_bonds, NULL, "BondBlob", NULL},
    { "bond_indices", sb_bond_indices, NULL,
        "Nx2 numpy array of indices into the corresponding AtomBlob", NULL},
    { "displays", sb_displays, sb_set_displays,
        "numpy array of (bool) displays", NULL},
    { "structures", sb_structures, NULL,
        "list of one-structure-model StructBlobs", NULL},
    { "residues", sb_residues, NULL, "ResBlob", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

} // extern "C"

PyTypeObject StructBlob_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "structaccess.StructBlob", // tp_name
    sizeof (StructBlob), // tp_basicsize
    0, // tp_itemsize
    StructBlob_dealloc, // tp_dealloc
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
    StructBlob_doc, // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    //sizeof(PyObject), // tp_weaklistoffset
    offsetof(StructBlob, _weaklist), // tp_weaklistoffset
    0, // tp_iter
    0, // tp_iternext
    StructBlob_methods, // tp_methods
    0, // tp_members
    StructBlob_getset, // tp_getset
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
