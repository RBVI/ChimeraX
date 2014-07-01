//#include <iostream>			// use std:cerr for debugging
#include <stdlib.h>			// use strncpy
#include <ctype.h>			// use isspace
#include <string.h>			// use memset
#include <vector>			// use std::vector

#include "parsepdb.h"			// use Atom
#include "pythonarray.h"		// use python_float_array
#include "stringnum.h"			// use string_to_float()

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

static void parse_pdb(const char *pdb, std::vector<Atom> &atoms, std::vector<int> &molstart)
{
  char buf[9];
  buf[8] = '\0';
  int ni, s;
  char last_alt_loc = ' ';
  size_t asize = sizeof(Atom);
  Atom atom;
  molstart.push_back(0);
  for (int i = 0 ; pdb[i] ; i = ni)
    {
      	const char *line = pdb + i;
	ni = next_line(pdb, i);
	int line_len = ni - i;
	if (strncmp(line, "ENDMDL", 6) == 0)
	  molstart.push_back(atoms.size());
	else if (atom_line(line) && line_len > 46)
	  {
	    memset(&atom, 0, asize);
	    strncpy(buf, line+30, 8);
	    atom.x = string_to_float(buf);
	    strncpy(buf, line+38, 8);
	    atom.y = string_to_float(buf);
	    strncpy(buf, line+46, 8);
	    atom.z = string_to_float(buf);
	    atom.chain_id[0] = line[21];
	    int e = (line_len > 76 ? element_number(line + 76) : 0);
	    atom.element_number = e;
	    strncpy(buf, line+22, 4);
	    atom.residue_number = string_to_integer(buf);
	    for (s = 17 ; s < 19 && line[s] == ' ' ; ++s) ;	// Skip leading spaces.
	    for (e = 19 ; e > 16 && line[e] == ' ' ; --e) ;	// Skip trailing spaces.
	    if (e >= s)
	      strncpy(atom.residue_name, line+s, e-s+1);
	    for (s = 12 ; s < 15 && line[s] == ' ' ; ++s) ;	// Skip leading spaces.
	    for (e = 15 ; e > 11 && line[e] == ' ' ; --e) ;	// Skip trailing spaces.
	    if (e >= s)
	      strncpy(atom.atom_name, line+s, e-s+1);
	    if (atoms.size() > molstart.back() && line[16] != last_alt_loc)
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
  if (molstart.back() >= atoms.size())
    molstart.pop_back();
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
  std::vector<int> molstart;
  parse_pdb(pdb_text, atoms, molstart);

  int nm = molstart.size();
  PyObject *mol_atoms = PyTuple_New(nm);
  size_t ta = atoms.size(), asize = sizeof(Atom);
  for (int m = 0 ; m < nm ; ++m)
    {
      char *adata;
      int ms = molstart[m];
      int na = (m < nm-1 ? molstart[m+1]-ms : ta-ms);
      PyObject *atoms_py = python_char_array(na, asize, &adata);
      memcpy(adata, atoms.data() + ms, na*asize);
      PyTuple_SetItem(mol_atoms, m, atoms_py);
    }

  return mol_atoms;
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
class Atom_Compare
{
public:
  Atom_Compare(Atom *atoms) : atoms(atoms) {}
  bool operator() (int i, int j)
  {
    const Atom *at1 = &atoms[i], *at2 = &atoms[j];
    int ccmp = strncmp(at1->chain_id, at2->chain_id, CHAIN_ID_LEN);
    if (ccmp == 0)
      {
	int r1 = at1->residue_number, r2 = at2->residue_number;
	ccmp = (r1 < r2 ? -1 : (r1 > r2 ? 1 : 0));
      }
    return ccmp < 0;
  }
private:
  Atom *atoms;
};

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
atom_sort_order(PyObject *s, PyObject *args, PyObject *keywds)
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
      PyErr_SetString(PyExc_TypeError, "atom_sort_order(): array must be 2 dimensional");
      return NULL;
    }
  if (!atoms.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "atom_sort_order(): array must be contiguous");
      return NULL;
    }

  std::vector<int> order(n);
  for (int i = 0 ; i < n ; ++i)
    order[i] = i;

  std::sort(order.begin(), order.end(), Atom_Compare(aa));

  return c_array_to_python(order);
}

// ----------------------------------------------------------------------------
// Calculate unique residue ids (same chain id and residue number).
// Assumes that the atoms are sorted by chain id and residues number.
//
static void residue_ids(const Atom *atoms, int n, int *rids)
{
  int rid = 0;
  const Atom *ar = atoms;
  for (int a = 0 ; a < n ; ++a)
    {
      const Atom *aa = &atoms[a];
      if (aa->residue_number != ar->residue_number || strncmp(aa->chain_id, ar->chain_id, CHAIN_ID_LEN))
	{
	  rid += 1;
	  ar = aa;
	}
      rids[a] = rid;
    }
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
residue_ids(PyObject *s, PyObject *args, PyObject *keywds)
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
      PyErr_SetString(PyExc_TypeError, "residue_ids(): array must be 2 dimensional");
      return NULL;
    }
  if (!atoms.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError, "residue_ids(): array must be contiguous");
      return NULL;
    }

  int *rids;
  PyObject *rids_py = python_int_array(n, &rids);
  residue_ids(aa, n, rids);

  return rids_py;
}
