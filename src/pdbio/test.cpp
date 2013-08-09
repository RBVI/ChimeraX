#include "pdbio/PDBio.h"
#include "molecule/Molecule.h"
#include "molecule/Bond.h"
#include "molecule/Atom.h"
#include "molecule/Coord.h"
#include "molecule/Residue.h"
#include <iostream>
#include <stdio.h>

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
	PyObject *capsule = read_pdb(py_file, NULL, true);
	int num_residues = 0;
	int num_atoms = 0;
	int num_bonds = 0;
	if (!PyCapsule_CheckExact(capsule)) {
		std::cerr << "Didn't return a capsule!\n";
		return -1;
	}
	FILE *f = fopen("pdbio.bild", "w");
	std::vector<Molecule *> *mols = (std::vector<Molecule *> *) PyCapsule_GetPointer(capsule, "pdbio.mol_vector");
	if (mols == NULL) {
		std::cerr << "Capsule didn't contain a vector of Molecules.\n";
		return -1;
	}
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		Molecule::Atoms atoms = m->atoms();
		Molecule::Bonds bonds = m->bonds();
		Molecule::Residues residues = m->residues();
#ifdef DEBUG_BONDS
		std::map<std::string, int> num_res_bonds;
		std::map<std::string, Residue *> exemplars;
		for (Molecule::Residues::iterator ri = residues.begin(); ri != residues.end();
		++ri) {
			Residue *r = *ri;
			int num_bonds = 0;
			Residue::Atoms r_atoms = r->atoms();
			for (Residue::Atoms::iterator rai = r_atoms.begin(); rai != r_atoms.end();
			++rai) {
				Atom *ra = *rai;
				Atom::Bonds a_bonds = ra->bonds();
				for (Atom::Bonds::iterator abi = a_bonds.begin(); abi != a_bonds.end();
				++abi) {
					Bond *b = *abi;
					if (b->atoms()[0]->residue() == r && b->atoms()[1]->residue() == r) {
						num_bonds++;
					}
				}
			}
			num_bonds /= 2;
			if (exemplars.find(r->name()) == exemplars.end()) {
				exemplars[r->name()] = r;
				num_res_bonds[r->name()] = num_bonds;
			} else {
				if (num_res_bonds[r->name()] != num_bonds) {
					std::cerr << "residue " << exemplars[r->name()]->str() << " has " <<
						num_res_bonds[r->name()] << " bonds whereas " <<
						r->str() << " has " << num_bonds << " bonds\n";
					return -1;
				}
			}
		}
#endif  // DEBUG_BONDS
		num_residues += residues.size();
		num_atoms += atoms.size();
		num_bonds += bonds.size();
		for (Molecule::Atoms::iterator ai = atoms.begin(); ai != atoms.end(); ++ai) {
			Atom *a = *ai;
			const char *color;
			float radius;
			if (strcmp(a->element().name(), "C") == 0) {
				color = "gray";
				radius = 0.07;
			} else if (strcmp(a->element().name(), "H") == 0) {
				color = "white";
				radius = 0.025;
			} else if (strcmp(a->element().name(), "O") == 0) {
				color = "red";
				radius = 0.06;
			} else if (strcmp(a->element().name(), "N") == 0) {
				color = "blue";
				radius = 0.065;
			} else if (strcmp(a->element().name(), "S") == 0) {
				color = "yellow";
				radius = 0.1;
			} else if (strcmp(a->element().name(), "P") == 0) {
				color = "orange";
				radius = 0.1;
			} else {
				color = "cyan";
				radius = 0.14;
			}
			radius *= 3.0;
			Coord crd = a->coord();
			fprintf(f, ".color %s\n", color);
			fprintf(f, ".sphere %g %g %g %g\n", crd[0], crd[1], crd[2], radius);
		}
		fprintf(f, ".color white\n");
		for (Molecule::Bonds::iterator bi = bonds.begin(); bi != bonds.end(); ++bi) {
			Bond *b = *bi;
			Atom *a1 = b->atoms()[0];
			Atom *a2 = b->atoms()[1];
			Coord crd1 = a1->coord();
			Coord crd2 = a2->coord();
			fprintf(f, ".cylinder %g %g %g %g %g %g 0.03 open\n", crd1[0], crd1[1], crd1[2],
				crd2[0], crd2[1], crd2[2]);
		}
	}
	fclose(f);
	std::cout << mols->size() << " molecules, " << num_residues << " residues, "
		<< num_bonds << " bonds, and " << num_atoms << " atoms\n";
	return 0;
}
