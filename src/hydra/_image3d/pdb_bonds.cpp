//
// Use reference data describing standard PDB chemical components
// to determine which atoms are bonded in a molecule.
//
//#include <iostream>		// use std::cerr
#include <map>			// use std::map
#include <set>			// use std::set
#include <string>		// use std::string
#include <vector>		// use std::vector
#include <string.h>		// use strnlen()
#include "pythonarray.h"	// use parse_int_n_array(), ...

typedef std::vector<int> Bond_List;		// Two atom indices per bond.
typedef std::vector<const char *> Index_Atom;		// Res template atom index to all_bonds index giving atom name.
typedef Reference_Counted_Array::Array<char> CArray;
typedef std::vector<std::string> String_List;

class Res_Template
{
public:
  void initialize_bond_list(const CArray &allbonds, int bi);
  void bonds(int *atom_num, int amin, Bond_List &bonds);
  bool atom_index(const char *aname, int alen, int *ai);
  Bond_List index_pairs;
  Index_Atom iatom;
  std::vector<int> common_order;
  int next;
};

class Bond_Templates
{
public:
  Bond_Templates(const IArray &cc_index, const CArray &all_bonds, const std::string &rname_letters);

  void molecule_bonds(const char *rnames, int rname_chars,
		      const int *rnums,
		      const char *cids, int cid_chars,
		      const char *anames, int aname_chars,
		      int natoms,
		      Bond_List &bonds,
		      String_List &missing_template);
  void backbone_bonds(const char *rnames, int rname_chars,
		      const int *rnums,
		      const char *cids, int cid_chars,
		      const char *anames, int aname_chars,
		      int natoms,
		      Bond_List &bonds);
  Res_Template *chemical_component_bond_table(const std::string &rname);
  int component_index(const std::string &rname);

private:
  IArray cc_index;       // Index into all bonds list for each chemical component
  CArray all_bonds;     // Bonds for all chemical components.
                        //   Array of atom names, each 4 characters, a pair for each bond,
                        //   empty name separates chemical components.
  std::string rname_letters; // Chemical component id can be only be in these letters.

  std::map<std::string, Res_Template> cc_bond_table;   // Bond table for each chemical component
};

Bond_Templates::Bond_Templates(const IArray &cc_index, const CArray &all_bonds, const std::string &rname_letters)
{
  this->cc_index = cc_index;
  this->all_bonds = all_bonds;
  this->rname_letters = rname_letters;
}

static std::string array_string(const char *a, int i, int size)
{
  const char *ai = a + i*size;
  return std::string(ai, strnlen(ai,size));
}
  
void Bond_Templates::molecule_bonds(const char *rnames, int rname_chars,
				    const int *rnums,
				    const char *cids, int cid_chars,
				    const char *anames, int aname_chars,
				    int natoms,
				    Bond_List &bonds,
				    String_List &missing_template)
{
  const char *res_rname = NULL, *res_cid = NULL;
  int res_rnum = 0;
  Res_Template *rtemplate = NULL;
  std::set<std::string> missing;
  int *atom_num = new int[1024];
  memset(atom_num, -1, 1024*sizeof(int));
  int amin = 0;
  for (int a = 0 ; a < natoms ; ++a)
    {
      const char *rname = rnames + rname_chars*a;
      int rnum = rnums[a];
      const char *cid = cids + cid_chars*a;
      bool same_res = (rnum == res_rnum &&
		       res_rname != NULL && strncmp(rname, res_rname, rname_chars) == 0 &&
		       res_cid != NULL && strncmp(cid, res_cid, cid_chars) == 0);
      if (! same_res)
	{
	  if (rtemplate)
	    rtemplate->bonds(atom_num, amin, bonds);
	  std::string srname = array_string(rnames, a, rname_chars);
	  rtemplate = chemical_component_bond_table(srname);
	  if (!rtemplate)
	    missing.insert(srname);
	  res_rname = rname;
	  res_cid = cid;
	  res_rnum = rnum;
	  amin = a;
	}

      const char *aname = anames + a*aname_chars;
      int ai;
      if (rtemplate && rtemplate->atom_index(aname, aname_chars, &ai))
      	atom_num[ai] = a;
      // else atom has no bonds, maybe non-standard atom name, or a single atom ion.
    }
  missing_template.insert(missing_template.end(), missing.begin(), missing.end());

  if (rtemplate)
    rtemplate->bonds(atom_num, amin, bonds);

  backbone_bonds(rnames, rname_chars, rnums, cids, cid_chars, anames, aname_chars, natoms, bonds);
}

void Res_Template::bonds(int *atom_num, int amin, Bond_List &bonds)
{
  int ip2 = index_pairs.size();
  for (int i = 0 ; i < ip2 ; i += 2)
    {
      int a1 = atom_num[index_pairs[i]];
      if (a1 >= amin)
	{
	  int a2 = atom_num[index_pairs[i+1]];
	  if (a2 >= amin)
	    {
	      bonds.push_back(a1);
	      bonds.push_back(a2);
	    }
	}
    }
}

void Res_Template::initialize_bond_list(const CArray &all_bonds, int bi)
{
  next = 0;
  if (bi == -1)
    return;

  const char *ab = all_bonds.values();
  int alen = all_bonds.size(1);
  int i1, i2;
  for ( ; ab[bi*alen] ; bi += 2)
    {
      const char *a1 = ab + bi*alen;
      if (!atom_index(a1, alen, &i1))
	{ i1 = iatom.size(); iatom.push_back(a1); }
      index_pairs.push_back(i1);
      const char *a2 = a1 + alen;
      if (!atom_index(a2, alen, &i2))
	{ i2 = iatom.size(); iatom.push_back(a2); }
      index_pairs.push_back(i2);
    }
  common_order.resize(iatom.size(), 0);
}

bool Res_Template::atom_index(const char *aname, int alen, int *ai)
{
  if (next < common_order.size())
    {
      int j = common_order[next];
      if (strncmp(iatom[j], aname, alen) == 0)
	{
	  *ai = j;
	  next = (next + 1) % common_order.size();
	  return true;
	}
    }
  int na = iatom.size();
  for (int i = 0 ; i < na ; ++i)
    if (strncmp(iatom[i], aname, alen) == 0)
      {
	*ai = i;
	if (next < common_order.size())
	  common_order[next++] = i;
	return true;
      }
  return false;
}

// Connect consecutive residues in proteins and nucleic acids.
void Bond_Templates::backbone_bonds(const char *rnames, int rname_chars,
				    const int *rnums,
				    const char *cids, int cid_chars,
				    const char *anames, int aname_chars,
				    int natoms,
				    Bond_List &bonds)
{
  // Assumes residues numbers are in increasing order within each chain and chains are contiguous.
  int a1 = -1, alink = -1;
  int cur_rnum = -1000000;
  const char *cur_cid = "";
  for (int a = 0 ; a < natoms ; ++a)
    {
      int rnum = rnums[a];
      const char *cid = cids + a*cid_chars;
      if (rnum != cur_rnum || strncmp(cid, cur_cid, cid_chars) != 0)
	{
	  alink = ((rnum == cur_rnum + 1 && strncmp(cid, cur_cid, cid_chars) == 0) ? a1 : -1);
	  cur_rnum = rnum;
	  cur_cid = cid;
	}
      const char *an = anames + a*aname_chars;
      if ((an[0] == 'C' && an[1] == '\0') ||
	  (an[0] == 'O' && an[1] == '3' && an[2] == '\'' && an[3] == '\0'))
	a1 = a;
      else if (alink >= 0 && ((an[0] == 'N' && an[1] == '\0') ||
			      (an[0] == 'P' && an[1] == '\0')))
	{
	  bonds.push_back(alink);
	  bonds.push_back(a);
	  alink = -1;
	}
    }
}

Res_Template *Bond_Templates::chemical_component_bond_table(const std::string &rname)
{
  std::map<std::string, Res_Template>::iterator i = cc_bond_table.find(rname);
  if (i != cc_bond_table.end())
    {
      Res_Template &rtemp = i->second;
      rtemp.next = 0;
      return &rtemp;
    }

  int ci = component_index(rname);
  if (ci == -1)
    return NULL;

  Res_Template &rtemp = cc_bond_table[rname];
  int bi = cc_index.values()[ci];
  rtemp.initialize_bond_list(all_bonds, bi);
  return &rtemp;
}

//
// Map every 3 character chemical component name to an integer.
//
int Bond_Templates::component_index(const std::string &rname)
{
  int n = rname_letters.size();
  int k = 0, j;
  int rsize = rname.size();
  for (int i = 0 ; i < rsize ; ++i)
    {
      char c = rname[i];
      for (j = 0 ; j < n && rname_letters[j] != c ; ++j) ;
      if (j == n)
	return -1;	// Illegal character.
      k = k*n + j;
    }
  return k;
}

static Bond_Templates *bond_templates = NULL;

// ----------------------------------------------------------------------------
// Return bonds derived from residue templates where each bond is a pair of atom numbers.
// Returned bonds are an N by 2 numpy array.
//
extern "C" PyObject *
initialize_bond_templates(PyObject *s, PyObject *args, PyObject *keywds)
{
  IArray cindex;
  Reference_Counted_Array::Array<char> all_bonds;
  const char *rlet;
  const char *kwlist[] = {"chemical_index", "all_bonds", "rname_letters", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&s"), (char **)kwlist,
				   parse_int_n_array, &cindex, parse_string_array, &all_bonds,
				   &rlet))
    return NULL;

  if (bond_templates)
    delete bond_templates;
  bond_templates = new Bond_Templates(cindex, all_bonds, rlet);

  Py_INCREF(Py_None);
  return Py_None;
}

// ----------------------------------------------------------------------------
// Return bonds derived from residue templates where each bond is a pair of atom numbers.
// Returned bonds are an N by 2 numpy array.
//
extern "C" PyObject *
molecule_bonds(PyObject *s, PyObject *args, PyObject *keywds)
{
  CArray rnames, cids, anames;
  IArray rnums;
  const char *kwlist[] = {"rnames", "rnums", "cids", "anames", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&"),
				   (char **)kwlist,
				   parse_string_array, &rnames,
				   parse_int_n_array, &rnums,
				   parse_string_array, &cids,
				   parse_string_array, &anames))
    return NULL;
  if (rnames.size(0) != anames.size(0) || rnums.size(0) != anames.size(0) || cids.size(0) != anames.size(0))
    {
      PyErr_SetString(PyExc_TypeError, "molecule_bonds: rnames, rnums, cids, anames must have same size");
      return NULL;
    }

  if (!bond_templates)
    {
      PyErr_SetString(PyExc_AssertionError, "Called molecule_bonds() before initialize_bond_templates()");
      return NULL;
    }

  Bond_List bonds;
  String_List missing;
  bond_templates->molecule_bonds(rnames.values(), rnames.size(1), rnums.values(),
				 cids.values(), cids.size(1),
				 anames.values(), anames.size(1), anames.size(0),
				 bonds, missing);

  int nb2 = bonds.size();
  int nb = nb2/2;
  int *ba;
  PyObject *bonds_py = python_int_array(nb, 2, &ba);
  for (int i = 0 ; i < nb2 ; i += 2)
    {
      ba[i] = bonds[i];
      ba[i+1] = bonds[i+1];
    }

  int nm = missing.size();
  PyObject *missing_py = PyTuple_New(nm);
  for (int i = 0 ; i < nm ; ++i)
    PyTuple_SetItem(missing_py, i, PyBytes_FromString(missing[i].c_str()));

  return python_tuple(bonds_py, missing_py);
}
