#include "capsule.h"
#include <stdio.h>
#include <stdexcept>
#include "molecule/Molecule.h"
#include "molecule/Residue.h"
#include "molecule/Bond.h"
#include "molecule/Atom.h"

static void
capsule_destructor(PyObject *capsule)
{
	const char *name = PyCapsule_GetName(capsule);
	void *ptr = PyCapsule_GetPointer(capsule, name);
	if (strcmp(name, "capsule.mol_vector")) {
		delete (std::vector<Molecule *> *)ptr;
	} else if (strcmp(name, "capsule.res_vector")) {
		delete (std::vector<Residue *> *)ptr;
	} else if (strcmp(name, "capsule.atom_vector")) {
		delete (std::vector<Atom *> *)ptr;
	} else if (strcmp(name, "capsule.bond_vector")) {
		delete (std::vector<Bond *> *)ptr;
	} else {
		throw std::invalid_argument("Don't recognize capsule type!");
	}
}

PyObject *
encapsulate_mol_vec(std::vector<Molecule *> *mols)
{
	return PyCapsule_New(mols, "capsule.mol_vector", capsule_destructor);
}

PyObject *
encapsulate_res_vec(std::vector<Residue *> *residues)
{
	return PyCapsule_New(residues, "capsule.res_vector", capsule_destructor);
}

PyObject *
encapsulate_atom_vec(std::vector<Atom *> *atoms)
{
	return PyCapsule_New(atoms, "capsule.atom_vector", capsule_destructor);
}

PyObject *
encapsulate_bond_vec(std::vector<Bond *> *bonds)
{
	return PyCapsule_New(bonds, "capsule.bond_vector", capsule_destructor);
}

std::vector<Molecule *> *
decapsulate_mol_vec(PyObject *capsule)
{
	if (!PyCapsule_CheckExact(capsule))
		throw std::invalid_argument("Not a capsule!");

	void *retval = PyCapsule_GetPointer(capsule, "capsule.mol_vector");
	if (retval == NULL) {
		PyErr_Clear();
		throw std::invalid_argument("Capsule does not contain mol vector!");
	}
	return (std::vector<Molecule *> *)retval;
}

std::vector<Residue *> *
decapsulate_res_vec(PyObject *capsule)
{
	if (!PyCapsule_CheckExact(capsule))
		throw std::invalid_argument("Not a capsule!");

	void *retval = PyCapsule_GetPointer(capsule, "capsule.res_vector");
	if (retval == NULL) {
		PyErr_Clear();
		throw std::invalid_argument("Capsule does not contain res vector!");
	}
	return (std::vector<Residue *> *)retval;
}

std::vector<Atom *> *
decapsulate_atom_vec(PyObject *capsule)
{
	if (!PyCapsule_CheckExact(capsule))
		throw std::invalid_argument("Not a capsule!");

	void *retval = PyCapsule_GetPointer(capsule, "capsule.atom_vector");
	if (retval == NULL) {
		PyErr_Clear();
		throw std::invalid_argument("Capsule does not contain atom vector!");
	}
	return (std::vector<Atom *> *)retval;
}

std::vector<Bond *> *
decapsulate_bond_vec(PyObject *capsule)
{
	if (!PyCapsule_CheckExact(capsule))
		throw std::invalid_argument("Not a capsule!");

	void *retval = PyCapsule_GetPointer(capsule, "capsule.bond_vector");
	if (retval == NULL) {
		PyErr_Clear();
		throw std::invalid_argument("Capsule does not contain bond vector!");
	}
	return (std::vector<Bond *> *)retval;
}
