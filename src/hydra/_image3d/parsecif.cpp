// Read mmCIF atom site fields with custom code.  Tried using cifparse-obj V7-1-00 (Sept 29, 2011)
// and it was 15 times slower and took 8 times the memory.

#include <iostream>
#include <ctype.h>			// use isspace()
#include <string.h>			// use memset()
#include <vector>

#include "parsepdb.h"			// use element_number(), Atom
#include "pythonarray.h"		// use python_float_array

class mmCIF_Atom
{
public:
  mmCIF_Atom()
  {
    memset(&atom, 0, sizeof(Atom));
    alt_loc = '\0';
    model_num = 1;
  }
  bool same_atom(const mmCIF_Atom &a) const
  {
    return (atom.residue_number == a.atom.residue_number &&
	    strncmp(atom.atom_name, a.atom.atom_name, ATOM_NAME_LEN) == 0 &&
	    strncmp(atom.residue_name, a.atom.residue_name, RESIDUE_NAME_LEN) == 0 &&
	    strncmp(atom.chain_id, a.atom.chain_id, CHAIN_ID_LEN) == 0);
  }
  Atom atom;
  char alt_loc;
  int model_num;
};

class Atom_Site_Columns
{
public:
  Atom_Site_Columns() : type_symbol(-1), label_atom_id(-1), label_alt_id(-1), label_comp_id(-1),
			label_asym_id(-1), Cartn_x(-1), Cartn_y(-1), Cartn_z(-1),
			model_num(-1), max_column(-1)
  {}
  bool found_all_columns()
  { return (type_symbol != -1 && label_atom_id != -1 && label_comp_id != -1 &&
	    label_asym_id != -1 && Cartn_x != -1 && Cartn_y != -1 && Cartn_z != -1 &&
	    max_column != -1);
  }
  int type_symbol;
  int label_atom_id;
  int label_alt_id;
  int label_comp_id;
  int label_asym_id;
  int label_seq_id;
  int Cartn_x;
  int Cartn_y;
  int Cartn_z;
  int model_num;
  int max_column;
};

static bool parse_mmcif_atoms(const char *buf, std::vector<Atom> &atoms);
static const char *parse_atom_site_column_positions(const char *buf, Atom_Site_Columns &f);
static bool parse_atom_site_line(const char *line, mmCIF_Atom &a, Atom_Site_Columns &f);

static const char *next_line(const char *line)
{
  while (*line != '\n' && *line != '\0')
    line += 1;
  return (*line == '\n' ? line+1 : line);
}

static bool parse_mmcif_atoms(const char *buf, std::vector<mmCIF_Atom> &atoms)
{
  Atom_Site_Columns fields;
  const char *line = parse_atom_site_column_positions(buf, fields);
  if (!fields.found_all_columns())
    {
      std::cerr << "Missing atom_site columns\n";
      return false;
    }
  for ( ; strncmp(line, "ATOM", 4) == 0 || strncmp(line, "HETATM", 6) == 0 ; line = next_line(line))
    {
      mmCIF_Atom a;
      parse_atom_site_line(line, a, fields);
      if (a.model_num > 1)
	break;	// TODO: Currently skip all but first model.
      if (atoms.size() > 0 &&
	  a.alt_loc != atoms.back().alt_loc &&
	  a.same_atom(atoms.back()))
	continue;	// TODO: Currently skipping alternate locations.
      atoms.push_back(a);
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
      else if (strncmp(colname, "label_alt_id", 12) == 0) { f.label_alt_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_comp_id", 13) == 0) { f.label_comp_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_asym_id", 13) == 0) { f.label_asym_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "label_seq_id", 12) == 0) { f.label_seq_id = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "Cartn_x", 7) == 0 && isspace(colname[7])) { f.Cartn_x = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "Cartn_y", 7) == 0 && isspace(colname[7])) { f.Cartn_y = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "Cartn_z", 7) == 0 && isspace(colname[7])) { f.Cartn_z = c; cmax = max(c,cmax); }
      else if (strncmp(colname, "pdbx_PDB_model_num", 18) == 0) { f.model_num = c; cmax = max(c,cmax); }
      c += 1;
      line = next_line(line);
    }
  f.max_column = cmax;
  return line;
}

static bool parse_atom_site_line(const char *line, mmCIF_Atom &a, Atom_Site_Columns &f)
{
  int c = 0;
  Atom &ad = a.atom;
  while (*line != '\n')
    {
      while (*line != ' ') line += 1;
      while (*line == ' ') line += 1;
      c += 1;
      int fl = 0;
      for (fl = 0 ; line[fl] != ' ' ; ++fl) ;
      if (c == f.type_symbol)
	ad.element_number = element_number(line);
      else if (c == f.label_atom_id)
	for (int i = 0 ; i < fl && i < ATOM_NAME_LEN ; ++i)
	  ad.atom_name[i] = line[i];
      else if (c == f.label_alt_id)
	a.alt_loc = line[0];
      else if (c == f.label_comp_id)
	for (int i = 0 ; i < fl && i < RESIDUE_NAME_LEN ; ++i)
	  ad.residue_name[i] = line[i];
      else if (c == f.label_asym_id)
	for (int i = 0 ; i < fl && i < CHAIN_ID_LEN ; ++i)
	  ad.chain_id[i] = line[i];
      else if (c == f.label_seq_id)
	ad.residue_number = strtol(line, NULL, 10);
      else if (c == f.Cartn_x)
	ad.x = strtof(line, NULL);
      else if (c == f.Cartn_y)
	ad.y = strtof(line, NULL);
      else if (c == f.Cartn_z)
	ad.z = strtof(line, NULL);
      else if (c == f.model_num)
	a.model_num = strtol(line, NULL, 10);
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

  std::vector<mmCIF_Atom> atoms;
  if (!parse_mmcif_atoms(mmcif_text, atoms))
    {
      PyErr_SetString(PyExc_ValueError, "parse_mmcif_file: error parsing file");
      return NULL;

    }

  size_t na = atoms.size(), asize = sizeof(Atom);
  char *adata;
  PyObject *atoms_py = python_char_array(na, asize, &adata);
  for (int i = 0 ; i < na ; ++i)
    memcpy(adata + i*asize, &atoms[i].atom, asize);

  return atoms_py;
}
