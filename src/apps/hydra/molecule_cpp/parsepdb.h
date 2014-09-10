#ifndef PARSEPDB_HEADER_INCLUDED
#define PARSEPDB_HEADER_INCLUDED

#include <string.h>			// use strncmp()
#include <Python.h>			// use PyObject

#define ATOM_NAME_LEN 4
#define RESIDUE_NAME_LEN 4
#define CHAIN_ID_LEN 4

typedef struct
{
  char atom_name[ATOM_NAME_LEN];
  float x,y,z;
  float radius;
  char residue_name[RESIDUE_NAME_LEN];
  int residue_number;
  char chain_id[CHAIN_ID_LEN];
  unsigned char atom_color[4];		// RGBA
  unsigned char ribbon_color[4];
  unsigned char element_number;
  char atom_shown;			// boolean 0/1
  char atom_style;			// 0=sphere, 1=stick, 2=ball&stick
  char ribbon_shown;			// boolean 0/1
  char atom_selected;			// boolean 0/1
} Atom;

inline bool compare_residues(const Atom &a, const Atom &b)
{
  int ccmp = strncmp(a.chain_id, b.chain_id, CHAIN_ID_LEN);
  return (ccmp == 0 ? a.residue_number < b.residue_number : ccmp < 0);
}


extern "C" {

PyObject *parse_pdb_file(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *element_radii(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *atom_sort_order(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *residue_ids(PyObject *s, PyObject *args, PyObject *keywds);

}

int element_number(const char *element_name);
const float *element_radius_array();

#endif
