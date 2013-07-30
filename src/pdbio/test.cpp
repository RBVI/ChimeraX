#include "pdbio/PDBio.h"
#include "molecule/Molecule.h"
#include "molecule/Bond.h"
#include "molecule/Atom.h"
#include <iostream>

int main(int argc, char **argv)
{
	if (argc == 1) {
		std::cerr << "please supply name of PDB file as first arg\n";
		return -1;
	}
	Py_Initialize();
	char buf[1000];
	PyObject *main_module = PyImport_AddModule("__main__");
	PyObject *globals = PyModule_GetDict(main_module);
	sprintf(buf, "open('%s')", argv[1]);
	PyObject *py_file = PyRun_String(buf, Py_eval_input, globals, globals);
	if (py_file == NULL) {
		std::cerr << "open() failed\n";
		return -1;
	}
	PyObject *mol_list = read_pdb(py_file, NULL, true);
	int num_residues = 0;
	int num_atoms = 0;
	int num_bonds = 0;
	if (!PyList_Check(mol_list)) {
		std::cerr << "Didn't return a list (of mols)!\n";
		return -1;
	}
	int num_mols = PyList_Size(mol_list);
	for (int mi = 0; mi < num_mols; ++mi) {
		PyObject *py_mol_capsule = PyList_GetItem(mol_list, mi);
		if (!PyCapsule_CheckExact(py_mol_capsule)) {
			std::cerr << "list item at position " << mi << " is not a capsule.\n";
			return -1;
		}
		Molecule *m = (Molecule *) PyCapsule_GetPointer(py_mol_capsule, NULL);
		num_residues += m->residues().size();
		num_atoms += m->atoms().size();
		num_bonds += m->bonds().size();
	}
	std::cout << num_mols << " molecules, " << num_residues << " residues, "
		<< num_bonds << " bonds, and " << num_atoms << " atoms\n";
	return 0;
}
