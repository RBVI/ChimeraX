#ifndef PARSEPDB_HEADER_INCLUDED
#define PARSEPDB_HEADER_INCLUDED

#include <Python.h>			// use PyObject

#define ATOM_NAME_LEN 4
#define RESIDUE_NAME_LEN 4
#define CHAIN_ID_LEN 4

typedef struct
{
  char atom_name[ATOM_NAME_LEN];
  int element_number;
  float x,y,z;
  float radius;
  char residue_name[RESIDUE_NAME_LEN];
  int residue_number;
  char chain_id[CHAIN_ID_LEN];
  unsigned char atom_color[4];		// RGBA
  unsigned char ribbon_color[4];
  char atom_shown;			// boolean 0/1
  char ribbon_shown;			// boolean 0/1
} Atom;

extern "C" {

PyObject *parse_pdb_file(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *element_radii(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *sort_atoms_by_chain(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *residue_ids(PyObject *s, PyObject *args, PyObject *keywds);

}

int element_number(const char *element_name);

#endif
