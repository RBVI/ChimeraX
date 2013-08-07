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
capsule_mol_vec(std::vector<Molecule *> *mols)
{
	return PyCapsule_New(mols, "capsule.mol_vector", capsule_destructor);
}

PyObject *
capsule_res_vec(std::vector<Residue *> *residues)
{
	return PyCapsule_New(residues, "capsule.res_vector", capsule_destructor);
}

PyObject *
capsule_atom_vec(std::vector<Atom *> *atoms)
{
	return PyCapsule_New(atoms, "capsule.atom_vector", capsule_destructor);
}

PyObject *
capsule_bond_vec(std::vector<Bond *> *bonds)
{
	return PyCapsule_New(bonds, "capsule.bond_vector", capsule_destructor);
}
