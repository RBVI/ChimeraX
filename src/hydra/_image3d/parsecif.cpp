// Read mmCIF atom site fields with custom code.  Tried using cifparse-obj V7-1-00 (Sept 29, 2011)
// and it was 15 times slower and took 8 times the memory.

#include <iostream>
#include <ctype.h>			// use isspace()
#include <string.h>			// use memset()
#include <vector>

#include "parsepdb.h"			// use element_number()
#include "pythonarray.h"		// use python_float_array

#define MAX_CHAR_ELEMENT_NAME 4
#define MAX_CHAR_ATOM_NAME 4
#define MAX_CHAR_RES_NAME 4
#define MAX_CHAR_CHAIN_ID 4

class Atom
{
public:
  Atom()
  {
    memset(atom_name, 0, MAX_CHAR_ELEMENT_NAME);
    memset(atom_name, 0, MAX_CHAR_ATOM_NAME);
    memset(residue_name, 0, MAX_CHAR_RES_NAME);
    memset(chain_id, 0, MAX_CHAR_CHAIN_ID);
  }
  char element_name[MAX_CHAR_ELEMENT_NAME];
  char atom_name[MAX_CHAR_ATOM_NAME];
  char residue_name[MAX_CHAR_RES_NAME];
  char chain_id[MAX_CHAR_CHAIN_ID];
  int residue_num;
  float x, y, z;
};

class Atom_Site_Columns
{
public:
  Atom_Site_Columns() : type_symbol(-1), label_atom_id(-1), label_comp_id(-1),
			label_asym_id(-1), Cartn_x(-1), Cartn_y(-1), Cartn_z(-1),
			max_column(-1)
  {}
  bool found_all_columns()
  { return (type_symbol != -1 && label_atom_id != -1 && label_comp_id != -1 &&
	    label_asym_id != -1 && Cartn_x != -1 && Cartn_y != -1 && Cartn_z != -1 &&
	    max_column != -1);
  }
  int type_symbol;
  int label_atom_id;
  int label_comp_id;
  int label_asym_id;
  int label_seq_id;
  int Cartn_x;
  int Cartn_y;
  int Cartn_z;
  int max_column;
};

static bool parse_mmcif_atoms(const char *buf, std::vector<Atom> &atoms);
static const char *parse_atom_site_column_positions(const char *buf, Atom_Site_Columns &f);
static bool parse_atom_site_line(const char *line, Atom &a, Atom_Site_Columns &f);

static const char *next_line(const char *line)
{
  while (*line != '\n' && *line != '\0')
    line += 1;
  return (*line == '\n' ? line+1 : line);
}

static bool parse_mmcif_atoms(const char *buf, std::vector<Atom> &atoms)
{
  Atom_Site_Columns fields;
  const char *line = parse_atom_site_column_positions(buf, fields);
  if (!fields.found_all_columns())
    {
      std::cerr << "Missing atom_site columns\n";
      return false;
    }
  while (strncmp(line, "ATOM", 4) == 0 || strncmp(line, "HETATM", 6) == 0)
    {
      Atom a;
      parse_atom_site_line(line, a, fields);
      atoms.push_back(a);
      line = next_line(line);
    }
  return true;
}

static int max(int a, int b)
{
  return (a > b ? a : b);
}

static const char *parse_atom_site_column_positions(const char *buf, Atom_Site_Columns &f)
{
  const char *line = buf;
  int c = 0, cmax = 0;
  while (strncmp(line, "_atom_site.", 11) != 0)
    line = next_line(line);
  while (strncmp(line, "_atom_site.", 11) == 0)
    {
      const char *colname = line + 11;
      if (strncmp(colname, "type_symbol", 11) == 0) { f.type_symbol = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_atom_id", 13) == 0) { f.label_atom_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_comp_id", 13) == 0) { f.label_comp_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_asym_id", 13) == 0) { f.label_asym_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_seq_id", 12) == 0) { f.label_seq_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "Cartn_x", 7) == 0 && isspace(colname[7])) { f.Cartn_x = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "Cartn_y", 7) == 0 && isspace(colname[7])) { f.Cartn_y = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "Cartn_z", 7) == 0 && isspace(colname[7])) { f.Cartn_z = c; cmax = max(c,cmax); }
      c += 1;
      line = next_line(line);
    }
  f.max_column = cmax;
  return line;
}

static bool parse_atom_site_line(const char *line, Atom &a, Atom_Site_Columns &f)
{
  int c = 0;
  while (*line != '\n')
    {
      while (*line != ' ') line += 1;
      while (*line == ' ') line += 1;
      c += 1;
      int fl = 0;
      for (fl = 0 ; line[fl] != ' ' ; ++fl) ;
      if (c == f.type_symbol)
	for (int i = 0 ; i < fl && i < MAX_CHAR_ELEMENT_NAME ; ++i)
	  a.element_name[i] = line[i];
      else if (c == f.label_atom_id)
	for (int i = 0 ; i < fl && i < MAX_CHAR_ATOM_NAME ; ++i)
	  a.atom_name[i] = line[i];
      else if (c == f.label_comp_id)
	for (int i = 0 ; i < fl && i < MAX_CHAR_RES_NAME ; ++i)
	  a.residue_name[i] = line[i];
      else if (c == f.label_asym_id)
	for (int i = 0 ; i < fl && i < MAX_CHAR_CHAIN_ID ; ++i)
	  a.chain_id[i] = line[i];
      else if (c == f.label_seq_id)
	a.residue_num = strtol(line, NULL, 10);
      else if (c == f.Cartn_x)
	a.x = strtof(line, NULL);
      else if (c == f.Cartn_y)
	a.y = strtof(line, NULL);
      else if (c == f.Cartn_z)
	a.z = strtof(line, NULL);
      if (c >= f.max_column)
	break;
    }
  return true;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
parse_mmcif_file(PyObject *s, PyObject *args, PyObject *keywds)
{
  const char *mmcif_text;
  const char *kwlist[] = {"text", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("s"),
				   (char **)kwlist, &mmcif_text))
    return NULL;

  std::vector<Atom> atoms;
  if (!parse_mmcif_atoms(mmcif_text, atoms))
    {
      PyErr_SetString(PyExc_ValueError, "parse_mmcif_file: error parsing file");
      return NULL;

    }

  int natom = atoms.size();
  float *xyz;
  int *residue_nums;
  unsigned char *element_nums;
  char *residue_names, *atom_names, *chain_ids;
  PyObject *xyz_py = python_float_array(natom, 3, &xyz);
  PyObject *element_nums_py = python_uint8_array(natom, &element_nums);
  PyObject *chain_ids_py = python_string_array(natom, MAX_CHAR_CHAIN_ID, &chain_ids);
  PyObject *residue_nums_py = python_int_array(natom, &residue_nums);
  PyObject *residue_names_py = python_string_array(natom, MAX_CHAR_RES_NAME, &residue_names);
  PyObject *atom_names_py = python_string_array(natom, MAX_CHAR_ATOM_NAME, &atom_names);

  for (int i = 0 ; i < natom ; ++i)
    {
      Atom &a = atoms[i];
      xyz[3*i] = a.x; xyz[3*i+1] = a.y; xyz[3*i+2] = a.z;
      residue_nums[i] = a.residue_num;
      memcpy(atom_names+MAX_CHAR_ATOM_NAME*i, a.atom_name, MAX_CHAR_ATOM_NAME);
      memcpy(residue_names+MAX_CHAR_RES_NAME*i, a.residue_name, MAX_CHAR_RES_NAME);
      memcpy(chain_ids+MAX_CHAR_CHAIN_ID*i, a.chain_id, MAX_CHAR_CHAIN_ID);
      element_nums[i] = element_number(a.element_name);
    }

  PyObject *t = PyTuple_New(6);
  PyTuple_SetItem(t, 0, xyz_py);
  PyTuple_SetItem(t, 1, element_nums_py);
  PyTuple_SetItem(t, 2, chain_ids_py);
  PyTuple_SetItem(t, 3, residue_nums_py);
  PyTuple_SetItem(t, 4, residue_names_py);
  PyTuple_SetItem(t, 5, atom_names_py);

  return t;
}
