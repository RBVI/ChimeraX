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

typedef std::pair<int,int> Bond;	// Pair of atom indices.
typedef std::vector<Bond> Bond_List;
typedef std::map<std::string,int> Atom_Indices;
typedef Reference_Counted_Array::Array<char> CArray;
typedef std::vector<std::string> String_List;

class Res_Template
{
public:
  void bonds(const std::map<int,int> &atom_num, Bond_List &bonds);
  bool atom_index(const std::string &aname, int *ai);
  Bond_List index_pairs;
  Atom_Indices aindex;
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
  int res_rnum = -1000000;
  Res_Template *rtemplate = NULL;
  std::map<int, int> atom_num;
  std::set<std::string> missing;
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
	    rtemplate->bonds(atom_num, bonds);
	  atom_num.clear();
	  std::string srname = array_string(rnames, a, rname_chars);
	  rtemplate = chemical_component_bond_table(srname);
	  if (!rtemplate)
	    missing.insert(srname);
	  res_rname = rname;
	  res_cid = cid;
	  res_rnum = rnum;
	}

      std::string aname = array_string(anames, a, aname_chars);
      int ai;
      if (rtemplate && rtemplate->atom_index(aname, &ai))
	atom_num[ai] = a;
      // else atom has no bonds, maybe non-standard atom name, or a single atom ion.
    }
  missing_template.insert(missing_template.end(), missing.begin(), missing.end());

  if (rtemplate)
    rtemplate->bonds(atom_num, bonds);

  backbone_bonds(rnames, rname_chars, rnums, cids, cid_chars, anames, aname_chars, natoms, bonds);
}

void Res_Template::bonds(const std::map<int,int> &atom_num, Bond_List &bonds)
{
  for (Bond_List::iterator i = index_pairs.begin() ; i != index_pairs.end() ; ++i)
    {
      int i1 = i->first, i2 = i->second;
      std::map<int,int>::const_iterator a1 = atom_num.find(i1);
      if (a1 != atom_num.end())
	{
	  std::map<int,int>::const_iterator a2 = atom_num.find(i2);
	  if (a2 != atom_num.end())
	    bonds.push_back(std::pair<int,int>(a1->second, a2->second));
	}
    }
}

bool Res_Template::atom_index(const std::string &aname, int *ai)
{
  Atom_Indices::iterator i = aindex.find(aname);
  if (i == aindex.end())
    return false;
  *ai = i->second;
  return true;
}

class Backbone_Atom
{
public:
  Backbone_Atom() {}
  Backbone_Atom(const Backbone_Atom &b) : rnum(b.rnum), cid(b.cid), aname(b.aname) {}
  Backbone_Atom(int rnum, const std::string &cid, const std::string &aname) : rnum(rnum), cid(cid), aname(aname) {}
  int rnum;
  const std::string cid;
  const std::string aname;
  bool operator<(const Backbone_Atom &b) const
    { return rnum < b.rnum || (rnum == b.rnum && (cid < b.cid || (cid == b.cid && aname < b.aname))); }
};

// Connect consecutive residues in proteins and nucleic acids.
void Bond_Templates::backbone_bonds(const char *rnames, int rname_chars,
				    const int *rnums,
				    const char *cids, int cid_chars,
				    const char *anames, int aname_chars,
				    int natoms,
				    Bond_List &bonds)
{
  std::map<Backbone_Atom,int> bbatoms;
  for (int a = 0 ; a < natoms ; ++a)
    {
      std::string aname = array_string(anames, a, aname_chars);
      if (aname == "C" || aname == "N" || aname == "O3'" || aname == "P")
	{
	  int rnum = rnums[a];
	  std::string cid = array_string(cids, a, cid_chars);
	  bbatoms[Backbone_Atom(rnum, cid, aname)] = a;
	}
    }
  for (std::map<Backbone_Atom,int>::iterator ba = bbatoms.begin() ; ba != bbatoms.end() ; ++ba)
    {
      const Backbone_Atom &bat = ba->first;
      if (bat.aname == "C")
	{
	  std::map<Backbone_Atom,int>::iterator ba2 = bbatoms.find(Backbone_Atom(bat.rnum+1, bat.cid, "N"));
	  if (ba2 != bbatoms.end())
	    bonds.push_back(std::pair<int,int>(ba->second,ba2->second));
	}
      else if (bat.aname == "O3'")
	{
	  std::map<Backbone_Atom,int>::iterator ba2 = bbatoms.find(Backbone_Atom(bat.rnum+1, bat.cid, "P"));
	  if (ba2 != bbatoms.end())
	    bonds.push_back(std::pair<int,int>(ba->second,ba2->second));
	}
    }
}

Res_Template *Bond_Templates::chemical_component_bond_table(const std::string &rname)
{
  std::map<std::string, Res_Template>::iterator i = cc_bond_table.find(rname);
  if (i != cc_bond_table.end())
    return &(i->second);

  int ci = component_index(rname);
  if (ci == -1)
    return NULL;

  Res_Template &rtemp = cc_bond_table[rname];
  Bond_List &ipairs = rtemp.index_pairs;
  Atom_Indices &aindex = rtemp.aindex;
  int bi = cc_index.values()[ci];
  char *ab = all_bonds.values();
  int alen = all_bonds.size(1);
  if (bi != -1)
    for ( ; ab[bi*alen] ; bi += 2)
      {
	std::string a1 = array_string(ab, bi, alen);
	Atom_Indices::iterator ai1 = aindex.find(a1);
	int i1 = (ai1 == aindex.end() ? (aindex[a1] = aindex.size()) : ai1->second);
	std::string a2 = array_string(ab, bi+1, alen);
	Atom_Indices::iterator ai2 = aindex.find(a2);
	int i2 = (ai2 == aindex.end() ? (aindex[a2] = aindex.size()) : ai2->second);
	ipairs.push_back(std::pair<int,int>(i1,i2)); 
      }
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

  int nb = bonds.size();
  int *ba;
  PyObject *bonds_py = python_int_array(nb, 2, &ba);
  int i = 0;
  for (Bond_List::iterator bi = bonds.begin() ; bi != bonds.end() ; ++bi, i += 2)
    {
      ba[i] = bi->first;
      ba[i+1] = bi->second;
    }

  int nm = missing.size();
  PyObject *missing_py = PyTuple_New(nm);
  for (i = 0 ; i < nm ; ++i)
    PyTuple_SetItem(missing_py, i, PyBytes_FromString(missing[i].c_str()));

  return python_tuple(bonds_py, missing_py);
}
