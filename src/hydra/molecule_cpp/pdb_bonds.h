#ifndef PDB_BONDS_HEADER_INCLUDED
#define PDB_BONDS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C" {

PyObject *molecule_bonds(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *initialize_bond_templates(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
