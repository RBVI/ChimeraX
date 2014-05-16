//#include <iostream>			// use std:cerr for debugging
#include <stdlib.h>			// use strtof, strncpy
#include <ctype.h>			// use isspace
#include <string.h>			// use memset
#include <vector>			// use std::vector

#include "parsepdb.h"			// use Atom
#include "pythonarray.h"		// use python_float_array

inline bool atom_line(const char *line)
{
  return strncmp(line, "ATOM", 4) == 0 || strncmp(line, "HETATM", 6) == 0;
}

inline int next_line(const char *s, int i)
{
  char c;
  for (c = s[i] ; c != '\n' && c != '\0' ; c = s[++i])
    ;
  if (c == '\n')
    i++;
  return i;
}

inline unsigned int element_hash(const char *element_name)
{
  unsigned char c0 = element_name[0], c1 = element_name[1];
  unsigned int e = (isspace(c0) ? 0 : (unsigned int) c0);
  if (c1 && !isspace(c1))
    e = e * 256 + c1;
  return e;
}

int element_number(const char *element_name)
{
  static int *elnum = NULL;
  if (elnum == NULL)
    {
      elnum = new int[65536];
      memset(elnum, 0, 65536*sizeof(int));
      const char *enames[] = {"H", "C", "N", "O", "P", "S", NULL};
      int enums[] = {1, 6, 7, 8, 15, 16};
      for (int i = 0 ; enames[i] ; ++i)
	{
	  unsigned int e = element_hash(enames[i]);
	  int n = enums[i];
	  elnum[e] = n;
	}
    }
  unsigned int e = element_hash(element_name);
  return elnum[e];
}

static void element_radii(int *element_nums, long n, long stride, float *radii)
{
  static float *erad = NULL;
  if (erad == NULL)
    {
      erad = new float[256];
      for (int i = 0 ; i < 256 ; ++i)
	erad[i] = 1.5;
      erad[1] = 1.00;	// H
      erad[6] = 1.70;	// C
      erad[7] = 1.625;	// N
      erad[8] = 1.50;	// O
      erad[15] = 1.871;	// P
      erad[16] = 1.782;	// S
    }
  for (int e = 0 ; e < n ; ++e)
    {
      int en = element_nums[e*stride];
      radii[e] = (en < 256 ? erad[en] : 0);
    }
}

static void parse_pdb(const char *pdb, std::vector<Atom> &atoms)
{
  char buf[9];
  buf[8] = '\0';
  int ni, s;
  char last_alt_loc = ' ';
  size_t asize = sizeof(Atom);
  Atom atom;
  for (int i = 0 ; pdb[i] ; i = ni)
    {
      	const char *line = pdb + i;
	ni = next_line(pdb, i);
	int line_len = ni - i;
	if (strncmp(line, "ENDMDL", 6) == 0)
	  return;	// Only parse the first model in the file.
	if (atom_line(line) && line_len > 46)
	  {
	    memset(&atom, 0, asize);
	    strncpy(buf, line+30, 8);
	    atom.x = strtof(buf, NULL);
	    strncpy(buf, line+38, 8);
	    atom.y = strtof(buf, NULL);
	    strncpy(buf, line+46, 8);
	    atom.z = strtof(buf, NULL);
	    atom.chain_id[0] = line[21];
	    int e = (line_len > 76 ? element_number(line + 76) : 0);
	    atom.element_number = e;
	    strncpy(buf, line+22, 4);
	    atom.residue_number = strtol(buf, NULL, 10);
	    for (s = 17 ; s < 19 && line[s] == ' ' ; ++s) ;	// Skip leading spaces.
	    for (e = 19 ; e > 16 && line[e] == ' ' ; --e) ;	// Skip trailing spaces.
	    if (e >= s)
	      strncpy(atom.residue_name, line+s, e-s+1);
	    for (s = 12 ; s < 15 && line[s] == ' ' ; ++s) ;	// Skip leading spaces.
	    for (e = 15 ; e > 11 && line[e] == ' ' ; --e) ;	// Skip trailing spaces.
	    if (e >= s)
	      strncpy(atom.atom_name, line+s, e-s+1);
	    if (!atoms.empty() && line[16] != last_alt_loc)
	      {
		const Atom &prev_atom = atoms.back();
		if (atom.residue_number == prev_atom.residue_number &&
		    strncmp(atom.atom_name, prev_atom.atom_name, ATOM_NAME_LEN) == 0 &&
		    strncmp(atom.residue_name, prev_atom.residue_name, RESIDUE_NAME_LEN) == 0 &&
		    atom.chain_id[0] == prev_atom.chain_id[0])
		  continue;		// Skip atom that differs only in alt-loc
	      }
	    last_alt_loc = line[16];
	    atoms.push_back(atom);
	  }
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
parse_pdb_file(PyObject *s, PyObject *args, PyObject *keywds)
{
  const char *pdb_text;
  const char *kwlist[] = {"text", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("y"),
				   (char **)kwlist, &pdb_text))
    return NULL;

  std::vector<Atom> atoms;
  parse_pdb(pdb_text, atoms);

  size_t na = atoms.size(), asize = sizeof(Atom);
  char *adata;
  PyObject *atoms_py = python_char_array(na, asize, &adata);
  memcpy(adata, atoms.data(), na*asize);

  return atoms_py;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
element_radii(PyObject *s, PyObject *args, PyObject *keywds)
{
  Numeric_Array elnum;
  const char *kwlist[] = {"element_numbers", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_int_n_array, &elnum))
    return NULL;
  int *el = (int *)elnum.values();
  long n = elnum.size(), st = elnum.stride(0);
  float *radii;
  PyObject *radii_py = python_float_array(n, &radii);
  element_radii(el, n, st, radii);

  return radii_py;
}

// ----------------------------------------------------------------------------
// Ordering function for chain id and residue number.
//
static int compare_atom_chains(const void *a1, const void *a2)
{
  const Atom *at1 = static_cast<const Atom *>(a1), *at2 = static_cast<const Atom *>(a2);
  int ccmp = strncmp(at1->chain_id, at2->chain_id, CHAIN_ID_LEN);
  if (ccmp == 0)
    {
      int r1 = at1->residue_number, r2 = at2->residue_number;
      ccmp = (r1 < r2 ? -1 : (r1 > r2 ? 1 : 0));
    }
  return ccmp;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
sort_atoms_by_chain(PyObject *s, PyObject *args, PyObject *keywds)
{
  CArray atoms;
  const char *kwlist[] = {"atoms", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_string_array, &atoms))
    return NULL;
  Atom *aa = (Atom *)atoms.values();
  int n = atoms.size(0), len = atoms.size(1);
  if (atoms.dimension() != 2)
    {
      PyErr_SetString(PyExc_TypeError, "sort_atoms_by_chain(): array must be 2 dimensional");
      return NULL;
    }
  if (!atoms.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "sort_atoms_by_chain(): array must be contiguous");
      return NULL;
    }
  qsort(aa, n, len, compare_atom_chains);

  Py_INCREF(Py_None);
  return Py_None;
}
