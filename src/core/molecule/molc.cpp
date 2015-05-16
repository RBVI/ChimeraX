#include <Python.h>	// Use PyUnicode_FromString

#include "atomstruct/Atom.h"
#include "atomstruct/Residue.h"
#include "blob/StructBlob.h"
#include "pythonarray.h"	// Use python_object_array()

//#include <iostream>

using namespace atomstruct;
using namespace basegeom;

extern "C" void atom_bfactor(void *atoms, int n, float *bfactors)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    bfactors[i] = a[i]->bfactor();
}

extern "C" void set_atom_bfactor(void *atoms, int n, float *bfactors)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    a[i]->set_bfactor(bfactors[i]);
}

extern "C" void atom_color(void *atoms, int n, unsigned char *rgba)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    {
      const Rgba &c = a[i]->color();
      *rgba++ = c.r;
      *rgba++ = c.g;
      *rgba++ = c.b;
      *rgba++ = c.a;
    }
}

extern "C" void set_atom_color(void *atoms, int n, unsigned char *rgba)
{
  Atom **a = static_cast<Atom **>(atoms);
  Rgba c;
  for (int i = 0 ; i < n ; ++i)
    {
      c.r = *rgba++;
      c.g = *rgba++;
      c.b = *rgba++;
      c.a = *rgba++;
      a[i]->set_color(c);
    }
}

extern "C" void atom_coord(void *atoms, int n, double *xyz)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    {
      const Coord &c = a[i]->coord();
      *xyz++ = c[0];
      *xyz++ = c[1];
      *xyz++ = c[2];
    }
}

extern "C" void set_atom_coord(void *atoms, int n, double *xyz)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    {
      Real x = *xyz++, y = *xyz++, z = *xyz++;
      a[i]->set_coord(Coord(x,y,z));
    }
}

extern "C" void atom_display(void *atoms, int n, unsigned char *disp)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    disp[i] = a[i]->display();
}

extern "C" void set_atom_display(void *atoms, int n, unsigned char *disp)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    a[i]->set_display(*disp++);
}

extern "C" void atom_draw_mode(void *atoms, int n, int *modes)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    modes[i] = a[i]->draw_mode();
}

extern "C" void set_atom_draw_mode(void *atoms, int n, int *modes)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    a[i]->set_draw_mode(modes[i]);
}

extern "C" void atom_element_name(void *atoms, int n, void **names)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    names[i] = PyUnicode_FromString(a[i]->element().name());
}

extern "C" void atom_element_number(void *atoms, int n, int *nums)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    nums[i] = a[i]->element().number();
}

extern "C" void atom_molecule(void *atoms, int n, void **molp)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    molp[i] = a[i]->structure();
}

extern "C" void atom_name(void *atoms, int n, void **names)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    names[i] = PyUnicode_FromString(a[i]->name());
}

extern "C" void atom_radius(void *atoms, int n, float *radii)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    *radii++ = a[i]->radius();
}

extern "C" void set_atom_radius(void *atoms, int n, float *radii)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    a[i]->set_radius(*radii++);
}

extern "C" void atom_residue(void *atoms, int n, void **resp)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    resp[i] = a[i]->residue();
}

extern "C" void bond_atoms(void *bonds, int n, void **atoms)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    {
      const Bond::Atoms &a = b[i]->atoms();
      *atoms++ = a[0];
      *atoms++ = a[1];
    }
}

extern "C" void bond_color(void *bonds, int n, unsigned char *rgba)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    {
      const Rgba &c = b[i]->color();
      *rgba++ = c.r;
      *rgba++ = c.g;
      *rgba++ = c.b;
      *rgba++ = c.a;
    }
}

extern "C" void set_bond_color(void *bonds, int n, unsigned char *rgba)
{
  Bond **b = static_cast<Bond **>(bonds);
  Rgba c;
  for (int i = 0 ; i < n ; ++i)
    {
      c.r = *rgba++;
      c.g = *rgba++;
      c.b = *rgba++;
      c.a = *rgba++;
      b[i]->set_color(c);
    }
}

extern "C" void bond_display(void *bonds, int n, int *disp)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    disp[i] = static_cast<int>(b[i]->display());
}

extern "C" void set_bond_display(void *bonds, int n, int *disp)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    b[i]->set_display(disp[i]);
}

extern "C" void bond_halfbond(void *bonds, int n, unsigned char *halfb)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    halfb[i] = b[i]->halfbond();
}

extern "C" void set_bond_halfbond(void *bonds, int n, unsigned char *halfb)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    b[i]->set_halfbond(halfb[i]);
}

extern "C" void bond_radius(void *bonds, int n, float *radii)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    *radii++ = b[i]->radius();
}

extern "C" void set_bond_radius(void *bonds, int n, float *radii)
{
  Bond **b = static_cast<Bond **>(bonds);
  for (int i = 0 ; i < n ; ++i)
    b[i]->set_radius(*radii++);
}

extern "C" void pseudobond_atoms(void *pbonds, int n, void **atoms)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    {
      const PBond::Atoms &a = b[i]->atoms();
      *atoms++ = a[0];
      *atoms++ = a[1];
    }
}

extern "C" void pseudobond_color(void *pbonds, int n, unsigned char *rgba)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    {
      const Rgba &c = b[i]->color();
      *rgba++ = c.r;
      *rgba++ = c.g;
      *rgba++ = c.b;
      *rgba++ = c.a;
    }
}

extern "C" void set_pseudobond_color(void *pbonds, int n, unsigned char *rgba)
{
  PBond **b = static_cast<PBond **>(pbonds);
  Rgba c;
  for (int i = 0 ; i < n ; ++i)
    {
      c.r = *rgba++;
      c.g = *rgba++;
      c.b = *rgba++;
      c.a = *rgba++;
      b[i]->set_color(c);
    }
}

extern "C" void pseudobond_display(void *pbonds, int n, int *disp)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    disp[i] = static_cast<int>(b[i]->display());
}

extern "C" void set_pseudobond_display(void *pbonds, int n, int *disp)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    b[i]->set_display(disp[i]);
}

extern "C" void pseudobond_halfbond(void *pbonds, int n, unsigned char *halfb)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    halfb[i] = b[i]->halfbond();
}

extern "C" void set_pseudobond_halfbond(void *pbonds, int n, unsigned char *halfb)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    b[i]->set_halfbond(halfb[i]);
}

extern "C" void pseudobond_radius(void *pbonds, int n, float *radii)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    *radii++ = b[i]->radius();
}

extern "C" void set_pseudobond_radius(void *pbonds, int n, float *radii)
{
  PBond **b = static_cast<PBond **>(pbonds);
  for (int i = 0 ; i < n ; ++i)
    b[i]->set_radius(*radii++);
}

extern "C" void residue_atoms(void *residues, int n, void **atoms)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    {
      const Residue::Atoms &a = r[i]->atoms();
      for (int j = 0 ; j < a.size() ; ++j)
	*atoms++ = a[i];
    }
}

extern "C" void residue_chain_id(void *residues, int n, void **cids)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    cids[i] = PyUnicode_FromString(r[i]->chain_id().c_str());
}

extern "C" void residue_molecule(void *residues, int n, void **molp)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    molp[i] = r[i]->structure();
}

extern "C" void residue_name(void *residues, int n, void **names)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    names[i] = PyUnicode_FromString(r[i]->name().c_str());
}

extern "C" void residue_num_atoms(void *residues, int n, int *natoms)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    natoms[i] = r[i]->atoms().size();
}

extern "C" void residue_number(void *residues, int n, int *nums)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    nums[i] = r[i]->position();
}

extern "C" void residue_str(void *residues, int n, void **strs)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    strs[i] = PyUnicode_FromString(r[i]->str().c_str());
}

extern "C" void residue_unique_id(void *residues, int n, int *rids)
{
  Residue **res = static_cast<Residue **>(residues);
  int rid = -1;
  const Residue *rprev = NULL;
  for (int i = 0 ; i < n ; ++i)
    {
      const Residue *r = res[i];
      if (rprev == NULL || r->position() != rprev->position() || r->chain_id() != rprev->chain_id()) {
	    rid += 1;
	    rprev = r;
	}
      rids[i] = rid;
    }
}

extern "C" void chain_chain_id(void *chains, int n, void **cids)
{
  Chain **c = static_cast<Chain **>(chains);
  for (int i = 0 ; i < n ; ++i)
    cids[i] = PyUnicode_FromString(c[i]->chain_id().c_str());
}

extern "C" void chain_molecule(void *chains, int n, void **molp)
{
  Chain **c = static_cast<Chain **>(chains);
  for (int i = 0 ; i < n ; ++i)
      *molp++ = c[i]->structure();
}

extern "C" void chain_num_residues(void *chains, int n, int *nres)
{
  Chain **c = static_cast<Chain **>(chains);
  for (int i = 0 ; i < n ; ++i)
    nres[i] = c[i]->residues().size();
}

extern "C" void chain_residues(void *chains, int n, void **res)
{
  Chain **c = static_cast<Chain **>(chains);
  for (int i = 0 ; i < n ; ++i)
    {
      const Chain::Residues &r = c[i]->residues();
      for (int j = 0 ; j < r.size() ; ++j)
	*res++ = r[i];
    }
}

extern "C" void molecule_name(void *mols, int n, void **names)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    names[i] = PyUnicode_FromString(m[i]->name().c_str());
}

extern "C" void set_molecule_name(void *mols, int n, void **names)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    m[i]->set_name(PyUnicode_AS_DATA(static_cast<PyObject *>(names[i])));
}

extern "C" void molecule_num_atoms(void *mols, int n, int *natoms)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    natoms[i] = m[i]->atoms().size();
}

extern "C" void molecule_atoms(void *mols, int n, void **atoms)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    {
      const AtomicStructure::Atoms &a = m[i]->atoms();
      for (int j = 0 ; j < a.size() ; ++j)
	*atoms++ = a[j].get();
    }
}

extern "C" void molecule_num_bonds(void *mols, int n, int *nbonds)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    nbonds[i] = m[i]->bonds().size();
}

extern "C" void molecule_bonds(void *mols, int n, void **bonds)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    {
      const AtomicStructure::Bonds &b = m[i]->bonds();
      for (int j = 0 ; j < b.size() ; ++j)
	*bonds++ = b[j].get();
    }
}

extern "C" void molecule_num_residues(void *mols, int n, int *nres)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    nres[i] = m[i]->residues().size();
}

extern "C" void molecule_residues(void *mols, int n, void **res)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    {
      const AtomicStructure::Residues &r = m[i]->residues();
      for (int j = 0 ; j < r.size() ; ++j)
	*res++ = r[j].get();
    }
}

extern "C" void molecule_num_chains(void *mols, int n, int *nchains)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    nchains[i] = m[i]->chains().size();
}

extern "C" void molecule_chains(void *mols, int n, void **chains)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    {
      const AtomicStructure::Chains &c = m[i]->chains();
      for (int j = 0 ; j < c.size() ; ++j)
	*chains++ = c[j].get();
    }
}

extern "C" void molecule_pbg_map(void *mols, int n, void **pbgs)
{
  // To use Python in this function which is called by ctypes,
  // must acquire the global interpreter lock.
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    {
      PyObject* pbg_map = PyDict_New();
      for (auto grp_info: m[i]->pb_mgr().group_map()) {
        PyObject* name = PyUnicode_FromString(grp_info.first.c_str());
	// Put these in numpy array: grp_info.second->pseudobonds() (type std::set<PBond*>)
	int np = grp_info.second->pseudobonds().size();
	void **pbga;
	PyObject *pb_array = python_voidp_array(np, &pbga);
	int p = 0;
        for (auto pb: grp_info.second->pseudobonds())
	  pbga[p++] = static_cast<void *>(pb);
	PyDict_SetItem(pbg_map, name, pb_array);
      }
      pbgs[i] = pbg_map;
    }

  PyGILState_Release(gstate);
}
