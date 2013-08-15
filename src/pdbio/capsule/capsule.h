#ifndef capsule_capsule
#define capsule_capsule

#include <Python.h>
#include <vector>

class Atom;
class Bond;
class Residue;
class Molecule;

PyObject * encapsulate_mol_vec(std::vector<Molecule *> *mols);
PyObject * encapsulate_res_vec(std::vector<Residue *> *residues);
PyObject * encapsulate_atom_vec(std::vector<Atom *> *atoms);
PyObject * encapsulate_bond_vec(std::vector<Bond *> *bonds);

std::vector<Molecule *> * decapsulate_mol_vec(PyObject *);
std::vector<Residue *> * decapsulate_res_vec(PyObject *);
std::vector<Atom *> * decapsulate_atom_vec(PyObject *);
std::vector<Bond *> * decapsulate_bond_vec(PyObject *);

#endif  // capsule_capsule
