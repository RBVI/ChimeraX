// vim: set expandtab ts=4 sw=4:
#include "MolBlob.h"
#include "AtomBlob.h"
#include "ResBlob.h"
#include "molecule/Bond.h"
#include <map>

extern "C" {
    
static void
MolBlob_dealloc(PyObject* obj)
{
    MolBlob* self = static_cast<MolBlob*>(obj);
    self->_items.clear();
    if (self->_weaklist)
        PyObject_ClearWeakRefs(obj);
    obj->ob_type->tp_free(obj);
}

static const char MolBlob_doc[] = "MolBlob documentation";

static int
MolBlob_init(PyObject *self, PyObject *args, PyObject *kw)
{
    char *kw_list[] = {NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kw, "", kw_list))
        return -1;
    MolBlob* molblob = static_cast<MolBlob*>(self);
    (void) new(molblob) MolBlob();
    return 0;
}

static PyObject*
mb_atoms(PyObject* self, void* null)
{
    PyObject* py_ab = PyObject_New(AtomBlob, &AtomBlob_type);
    py_ab = PyObject_Init(py_ab, &AtomBlob_type);
    AtomBlob* ab = static_cast<AtomBlob*>(py_ab);
    MolBlob* mb = static_cast<MolBlob*>(self);
    for (auto mi = mb->_items.begin(); mi != mb->_items.end(); ++mi) {
        const Molecule::Atoms& atoms = (*mi).get()->atoms();
        for (auto ai = atoms.begin(); ai != atoms.end(); ++ai)
            ab->_items.emplace_back((*ai).get());
    }
    return py_ab;
}

static PyObject*
mb_atoms_bonds(PyObject* self, void* null)
{
    MolBlob* mb = static_cast<MolBlob*>(self);
    PyObject* py_ab = mb_atoms(self, nullptr);
    AtomBlob* ab = static_cast<AtomBlob*>(py_ab);
    std::map<Atom *, AtomBlob::ItemsType::size_type> atom_map;
    decltype(atom_map)::mapped_type i = 0;
    auto& a_items = ab->_items;
    for (auto ai = a_items.begin(); ai != a_items.end(); ++ai, ++i) {
        atom_map[(*ai).get()] = i;
    }
    Molecule::Bonds::size_type bonds_size = 0;
    auto& m_items = mb->_items;
    for (auto mi = m_items.begin(); mi != m_items.end(); ++mi) {
        Molecule *m = (*mi).get();
        bonds_size += m->bonds().size();
    }
    i = 0;
    PyObject *bond_list = PyList_New(bonds_size);
    for (auto mi = m_items.begin(); mi != m_items.end(); ++mi) {
        Molecule *m = (*mi).get();
        auto& m_bonds = m->bonds();
        for (auto bi = m_bonds.begin(); bi != m_bonds.end(); ++bi) {
            Bond *b = (*bi).get();
            auto& b_atoms = b->atoms();
            PyObject *index_tuple = PyTuple_New(2);
            PyList_SET_ITEM(bond_list, i++, index_tuple);
            PyTuple_SET_ITEM(index_tuple, 0,
                PyLong_FromLong(atom_map[b_atoms[0]]));
            PyTuple_SET_ITEM(index_tuple, 1,
                PyLong_FromLong(atom_map[b_atoms[1]]));
        }
    }
    PyObject *ret_val = PyTuple_New(2);
    PyTuple_SET_ITEM(ret_val, 0, py_ab);
    PyTuple_SET_ITEM(ret_val, 1, bond_list);
    return ret_val;
}

static PyObject*
mb_molecules(PyObject* self, void* null)
{
    MolBlob* mb = static_cast<MolBlob*>(self);
    PyObject *mol_list = PyList_New(mb->_items.size());
    MolBlob::ItemsType::size_type i = 0;
    for (auto mi = mb->_items.begin(); mi != mb->_items.end(); ++mi) {
        PyObject* py_single_mb = PyObject_New(MolBlob, &MolBlob_type);
        PyList_SET_ITEM(mol_list, i++, py_single_mb);
        MolBlob* single_mb = static_cast<MolBlob*>(py_single_mb);
        single_mb->_items.emplace_back((*mi).get());
    }
    return mol_list;
}

static PyObject*
mb_residues(PyObject* self, void* null)
{
    PyObject* py_rb = PyObject_New(ResBlob, &ResBlob_type);
    py_rb = PyObject_Init(py_rb, &ResBlob_type);
    ResBlob* rb = static_cast<ResBlob*>(py_rb);
    MolBlob* mb = static_cast<MolBlob*>(self);
    for (auto mi = mb->_items.begin(); mi != mb->_items.end(); ++mi) {
        const Molecule::Residues& residues = (*mi).get()->residues();
        for (auto ri = residues.begin(); ri != residues.end(); ++ri)
            rb->_items.emplace_back((*ri).get());
    }
    return py_rb;
}

static PyMethodDef MolBlob_methods[] = {
    { NULL, NULL, 0, NULL }
};

static PyGetSetDef MolBlob_getset[] = {
    { "atoms", mb_atoms, NULL, "atom blob", NULL},
    { "atoms_bonds", mb_atoms_bonds, NULL,
        "2-tuple of (atom blob, list of atom-index 2-tuples)", NULL},
    { "molecules", mb_molecules, NULL,
        "list of one-molecule-model  molecule blobs", NULL},
    { "residues", mb_residues, NULL, "residue blob", NULL},
    { NULL, NULL, NULL, NULL, NULL }
};

} // extern "C"

PyTypeObject MolBlob_type = {
    PyObject_HEAD_INIT(NULL)
    "access.MolBlob", // tp_name
    sizeof (MolBlob), // tp_basicsize
    0, // tp_itemsize
    MolBlob_dealloc, // tp_dealloc
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
    MolBlob_doc, // tp_doc
    0, // tp_traverse
    0, // tp_clear
    0, // tp_richcompare
    offsetof(MolBlob, _weaklist), // tp_weaklist
    0, // tp_iter
    0, // tp_iternext
    MolBlob_methods, // tp_methods
    0, // tp_members
    MolBlob_getset, // tp_getset
    0, // tp_base
    0, // tp_dict
    0, // tp_descr_get
    0, // tp_descr_set
    0, // tp_dict_offset
    MolBlob_init, // tp_init,
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
