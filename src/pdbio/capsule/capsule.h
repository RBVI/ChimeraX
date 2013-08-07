#ifndef capsule_capsule
#define capsule_capsule

#include <Python.h>
#include <vector>

class Atom;
class Bond;
class Residue;
class Molecule;

PyObject * capsule_mol_vec(std::vector<Molecule *> *mols);

PyObject * capsule_res_vec(std::vector<Residue *> *residues);

PyObject * capsule_atom_vec(std::vector<Atom *> *atoms);

PyObject * capsule_bond_vec(std::vector<Bond *> *bonds);

#endif  // capsule_capsule
