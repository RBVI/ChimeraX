//#include <iostream>			// use std:cerr for debugging
#include <stdlib.h>			// use strtof, strncpy
#include <ctype.h>			// use isspace
#include <string.h>			// use memset

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

static int count_pdb_atoms(const char *pdb)
{
  int count = 0;
  for (int i = 0 ; pdb[i] ; i = next_line(pdb, i))
    if (atom_line(pdb+i))
      count += 1;
    else if (strncmp(pdb+i, "ENDMDL", 6) == 0)
      break;
  return count;
}

static unsigned int element_hash(const char *element_name)
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

static void element_radii(unsigned char *element_nums, int n, float *radii)
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
    radii[e] = erad[element_nums[e]];
}

static int parse_pdb(const char *pdb, int natom,
		     float *xyz, unsigned char *element_nums, char *chain_ids,
		     int *residue_nums, char *residue_names, char *atom_names)
{
  char buf[9];
  buf[8] = '\0';
  int a = 0, ni, s, e;
  for (int i = 0 ; pdb[i] && a < natom ; i = ni)
    {
      	const char *line = pdb + i;
	ni = next_line(pdb, i);
	int line_len = ni - i;
	if (strncmp(line, "ENDMDL", 6) == 0)
	  return a;	// Only parse the first model in the file.
	if (atom_line(line) && line_len > 46)
	  {
	    int a3 = 3*a;
	    strncpy(buf, line+30, 8);
	    xyz[a3] = strtof(buf, NULL);
	    strncpy(buf, line+38, 8);
	    xyz[a3+1] = strtof(buf, NULL);
	    strncpy(buf, line+46, 8);
	    xyz[a3+2] = strtof(buf, NULL);
	    chain_ids[a] = line[21];
	    int e = (line_len > 76 ? element_number(line + 76) : 0);
	    element_nums[a] = e;
	    strncpy(buf, line+22, 4);
	    residue_nums[a] = strtol(buf, NULL, 10);
	    for (s = 17 ; s < 19 && line[s] == ' ' ; ++s) ;	// Skip leading spaces.
	    for (e = 19 ; e > 16 && line[e] == ' ' ; --e) ;	// Skip trailing spaces.
	    if (e >= s)
	      strncpy(residue_names + 3*a, line+s, e-s+1);
	    for (s = 12 ; s < 15 && line[s] == ' ' ; ++s) ;	// Skip leading spaces.
	    for (e = 15 ; e > 11 && line[e] == ' ' ; --e) ;	// Skip trailing spaces.
	    if (e >= s)
	      strncpy(atom_names + 4*a, line+s, e-s+1);
	    a += 1;
	  }
    }
  return a;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
parse_pdb_file(PyObject *s, PyObject *args, PyObject *keywds)
{
  const char *pdb_text;
  const char *kwlist[] = {"text", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("s"),
				   (char **)kwlist, &pdb_text))
    return NULL;

  int natom = count_pdb_atoms(pdb_text);
  float *xyz;
  int *residue_nums;
  unsigned char *element_nums;
  char *chain_ids, *residue_names, *atom_names;
  PyObject *xyz_py = python_float_array(natom, 3, &xyz);
  PyObject *element_nums_py = python_uint8_array(natom, &element_nums);
  PyObject *chain_ids_py = python_string_array(natom, 1, &chain_ids);
  PyObject *residue_nums_py = python_int_array(natom, &residue_nums);
  PyObject *residue_names_py = python_string_array(natom, 3, &residue_names);
  memset(residue_names, 0, 3*natom);
  PyObject *atom_names_py = python_string_array(natom, 4, &atom_names);
  memset(atom_names, 0, 4*natom);

  parse_pdb(pdb_text, natom, xyz, element_nums, chain_ids,
	    residue_nums, residue_names, atom_names);

  PyObject *t = PyTuple_New(6);
  PyTuple_SetItem(t, 0, xyz_py);
  PyTuple_SetItem(t, 1, element_nums_py);
  PyTuple_SetItem(t, 2, chain_ids_py);
  PyTuple_SetItem(t, 3, residue_nums_py);
  PyTuple_SetItem(t, 4, residue_names_py);
  PyTuple_SetItem(t, 5, atom_names_py);

  return t;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
element_radii(PyObject *s, PyObject *args, PyObject *keywds)
{
  PyObject *elnum;
  const char *kwlist[] = {"element_numbers", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O"),
				   (char **)kwlist, &elnum))
    return NULL;

  Numeric_Array e = array_from_python(elnum, 1, Numeric_Array::Unsigned_Char, false);
  unsigned char *el = (unsigned char *)e.values();
  int n = e.size();
  float *radii;
  PyObject *radii_py = python_float_array(n, &radii);
  element_radii(el, n, radii);

  return radii_py;
}
