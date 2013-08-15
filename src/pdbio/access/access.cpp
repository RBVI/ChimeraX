#include "capsule/capsule.h"
#include "molecule/Molecule.h"
#include "molecule/Residue.h"
#include "molecule/Atom.h"
#include "molecule/Bond.h"
#include "molecule/Coord.h"
#include "molecule/Element.h"
#include <vector>
#include <map>
#include <Python.h>
#include <stdexcept>

static PyObject *
atoms(PyObject *, PyObject *mol_vec_capsule)
{
	std::vector<Molecule *> *mols;
	try {
		mols = decapsulate_mol_vec(mol_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms arg is not a mol-vec capsule");
		return NULL;
	}
	int size = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		size += m->atoms().size();
	}
	std::vector<Atom *> *atom_vec = new std::vector<Atom *>(0);
	atom_vec->reserve(size);
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Atoms &m_atoms = m->atoms();
		atom_vec->insert(atom_vec->end(), m_atoms.begin(), m_atoms.end());
	}
	return encapsulate_atom_vec(atom_vec);
}

// atoms_bonds: returns an atom blob and a list of 2-tuples
static PyObject *
atoms_bonds(PyObject *, PyObject *mol_vec_capsule)
{
	std::vector<Molecule *> *mols;
	try {
		mols = decapsulate_mol_vec(mol_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms_bonds arg is not a mol-vec capsule");
		return NULL;
	}
	int atoms_size = 0, bonds_size = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		atoms_size += m->atoms().size();
		bonds_size += m->bonds().size();
	}
	std::vector<Atom *> *atom_vec = new std::vector<Atom *>(0);
	atom_vec->reserve(atoms_size);
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Atoms &m_atoms = m->atoms();
		atom_vec->insert(atom_vec->end(), m_atoms.begin(), m_atoms.end());
	}
	std::map<Atom *, unsigned long> atom_map;
	unsigned long i = 0;
	for (std::vector<Atom *>::iterator ai = atom_vec->begin();
	ai != atom_vec->end(); ++ai, ++i) {
		atom_map[*ai] = i;
	}
	PyObject *bond_list = PyList_New(bonds_size);
	i = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Bonds &m_bonds = m->bonds();
		for (Molecule::Bonds::const_iterator bi = m_bonds.begin(); bi != m_bonds.end();
		++bi) {
			Bond *b = *bi;
			const Bond::Atoms &b_atoms = b->atoms();
			PyObject *index_tuple = PyTuple_New(2);
			PyList_SET_ITEM(bond_list, i++, index_tuple);
			PyTuple_SET_ITEM(index_tuple, 0, PyLong_FromLong(atom_map[b_atoms[0]]));
			PyTuple_SET_ITEM(index_tuple, 1, PyLong_FromLong(atom_map[b_atoms[1]]));
		}
	}
	PyObject *ret_val = PyTuple_New(2);
	PyTuple_SET_ITEM(ret_val, 0, encapsulate_atom_vec(atom_vec));
	PyTuple_SET_ITEM(ret_val, 1, bond_list);
	return ret_val;
}

static PyObject *
coords(PyObject *, PyObject *atom_vec_capsule)
{
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "coords arg is not a atom-vec capsule");
		return NULL;
	}
	PyObject *coords = PyList_New(atoms->size());
	unsigned long i = 0;
	for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end(); ++ai, ++i) {
		const Coord &crd = (*ai)->coord();
		PyObject *crd_tuple = PyTuple_New(3);
		PyList_SET_ITEM(coords, i, crd_tuple);
		PyTuple_SET_ITEM(crd_tuple, 0, PyFloat_FromDouble(crd[0]));
		PyTuple_SET_ITEM(crd_tuple, 1, PyFloat_FromDouble(crd[1]));
		PyTuple_SET_ITEM(crd_tuple, 2, PyFloat_FromDouble(crd[2]));
	}
	return coords;
}

static PyObject *
element_names(PyObject *, PyObject *atom_vec_capsule)
{
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "coords arg is not a atom-vec capsule");
		return NULL;
	}
	PyObject *element_names = PyList_New(atoms->size());
	unsigned long i = 0;
	for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end(); ++ai, ++i) {
		Element element = (*ai)->element();
		PyList_SET_ITEM(element_names, i, PyUnicode_FromString(element.name()));
	}
	return element_names;
}

static PyObject *
element_numbers(PyObject *, PyObject *atom_vec_capsule)
{
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "coords arg is not a atom-vec capsule");
		return NULL;
	}
	PyObject *element_numbers = PyList_New(atoms->size());
	unsigned long i = 0;
	for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end(); ++ai, ++i) {
		Element element = (*ai)->element();
		PyList_SET_ITEM(element_numbers, i, PyLong_FromLong((long)element.number()));
	}
	return element_numbers;
}

static struct PyMethodDef access_functions[] =
{
	{ "atoms", (PyCFunction)atoms, METH_O, "" },
	{ "atoms_bonds", (PyCFunction)atoms_bonds, METH_O, "" },
	{ "coords", (PyCFunction)coords, METH_O, "" },
	{ "element_names", (PyCFunction)element_names, METH_O, "" },
	{ "element_numbers", (PyCFunction)element_numbers, METH_O, "" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef access_def =
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
	return PyModule_Create(&access_def);
}
