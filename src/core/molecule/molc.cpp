#include <Python.h>	// Use PyUnicode_FromString

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>      // use PyArray_*(), NPY_*

#include "atomstruct/Atom.h"
#include "atomstruct/Chain.h"
#include "atomstruct/Pseudobond.h"
#include "atomstruct/Residue.h"
#include "basegeom/destruct.h"		// Use DestructionObserver
#include "pythonarray.h"		// Use python_voidp_array()

#include <iostream>

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

extern "C" void atom_bonds(void *atoms, int n, void **bonds)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    {
      const Atom::Bonds &b = a[i]->bonds();
      for (int j = 0 ; j < b.size() ; ++j)
	*bonds++ = b[j];
    }
}

extern "C" void atom_bonded_atoms(void *atoms, int n, void **batoms)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    {
      const Atom::Bonds &b = a[i]->bonds();
      for (int j = 0 ; j < b.size() ; ++j)
	*batoms++ = b[j]->other_atom(a[i]);
    }
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

extern "C" int atom_connects_to(void *atom, void *atom2)
{
  Atom *a = static_cast<Atom *>(atom), *a2 = static_cast<Atom *>(atom2);
  return (a->connects_to(a2) ? 1 : 0);
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

extern "C" void atom_delete(void *atoms, int n)
{
  Atom **a = static_cast<Atom **>(atoms);

  // Copy because deleting atoms modifies the input array.
  Atom **acopy = new Atom*[n];
  for (int i = 0 ; i < n ; ++i)
    acopy[i] = a[i];
  
  for (int i = 0 ; i < n ; ++i)
    acopy[i]->structure()->delete_atom(acopy[i]);

  delete [] acopy;
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

extern "C" void atom_num_bonds(void *atoms, int n, int *nbonds)
{
  Atom **a = static_cast<Atom **>(atoms);
  for (int i = 0 ; i < n ; ++i)
    nbonds[i] = a[i]->bonds().size();
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

extern "C" void *pseudobond_group_get(const char *name)
{
  int create = PBManager::GRP_NORMAL;	// Create if not yet created.
  PBGroup *pbg = PBManager::manager().get_group(name, create);
  return pbg;
}

extern "C" void pseudobond_group_delete(void *pbgroup)
{
  PBGroup *pbg = static_cast<PBGroup *>(pbgroup);
  delete pbg;
}

extern "C" void *pseudobond_group_new_pseudobond(void *pbgroup, void *atom1, void *atom2)
{
  PBGroup *pbg = static_cast<PBGroup *>(pbgroup);
  PBond *b = pbg->new_pseudobond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2));
  return b;
}

extern "C" void pseudobond_group_num_pseudobonds(void *pbgroups, int n, int *num_pseudobonds)
{
  PBGroup **pbg = static_cast<PBGroup **>(pbgroups);
  for (int i = 0 ; i < n ; ++i)
    *num_pseudobonds++ = pbg[i]->pseudobonds().size();
}

extern "C" void pseudobond_group_pseudobonds(void *pbgroups, int n, void **pseudobonds)
{
  PBGroup **pbg = static_cast<PBGroup **>(pbgroups);
  for (int i = 0 ; i < n ; ++i)
    for (auto pb: pbg[i]->pseudobonds())
      *pseudobonds++ = pb;
}

extern "C" void residue_atoms(void *residues, int n, void **atoms)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    {
      const Residue::Atoms &a = r[i]->atoms();
      for (int j = 0 ; j < a.size() ; ++j)
	*atoms++ = a[j];
    }
}

extern "C" void residue_chain_id(void *residues, int n, void **cids)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    cids[i] = PyUnicode_FromString(r[i]->chain_id().c_str());
}

extern "C" void residue_is_helix(void *residues, int n, unsigned char *is_helix)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
	  is_helix[i] = r[i]->is_helix();
}

extern "C" void set_residue_is_helix(void *residues, int n, unsigned char *is_helix)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
	  r[i]->set_is_helix(is_helix[i]);
}

extern "C" void residue_is_sheet(void *residues, int n, unsigned char *is_sheet)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    is_sheet[i] = r[i]->is_sheet();
}

extern "C" void set_residue_is_sheet(void *residues, int n, unsigned char *is_sheet)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    r[i]->set_is_sheet(is_sheet[i]);
}

extern "C" void residue_ss_id(void *residues, int n, int *ss_id)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    ss_id[i] = r[i]->ss_id();
}

extern "C" void set_residue_ss_id(void *residues, int n, int *ss_id)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    r[i]->set_ss_id(ss_id[i]);
}

extern "C" void residue_ribbon_display(void *residues, int n, unsigned char *ribbon_display)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    ribbon_display[i] = r[i]->ribbon_display();
}

extern "C" void set_residue_ribbon_display(void *residues, int n, unsigned char *ribbon_display)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    r[i]->set_ribbon_display(ribbon_display[i]);
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

extern "C" void residue_add_atom(void *res, void *atom)
{
  Residue *r = static_cast<Residue *>(res);
  r->add_atom(static_cast<Atom *>(atom));
}

extern "C" void residue_ribbon_color(void *residues, int n, unsigned char *rgba)
{
  Residue **r = static_cast<Residue **>(residues);
  for (int i = 0 ; i < n ; ++i)
    {
      const Rgba &c = r[i]->ribbon_color();
      *rgba++ = c.r;
      *rgba++ = c.g;
      *rgba++ = c.b;
      *rgba++ = c.a;
    }
}

extern "C" void set_residue_ribbon_color(void *residues, int n, unsigned char *rgba)
{
  Residue **r = static_cast<Residue **>(residues);
  Rgba c;
  for (int i = 0 ; i < n ; ++i)
    {
      c.r = *rgba++;
      c.g = *rgba++;
      c.b = *rgba++;
      c.a = *rgba++;
      r[i]->set_ribbon_color(c);
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

extern "C" void molecule_num_coord_sets(void *mols, int n, int *ncoord_sets)
{
  AtomicStructure **m = static_cast<AtomicStructure **>(mols);
  for (int i = 0 ; i < n ; ++i)
    ncoord_sets[i] = m[i]->coord_sets().size();
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

extern "C" PyObject *molecule_polymers(void *mol, int consider_missing_structure, int consider_chains_ids)
{
  // To use Python in this function which is called by ctypes,
  // must acquire the global interpreter lock.
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  AtomicStructure *m = static_cast<AtomicStructure *>(mol);
  std::vector<Chain::Residues> polymers = m->polymers(consider_missing_structure, consider_chains_ids);
  PyObject *poly = PyTuple_New(polymers.size());
  int p = 0;
  for (auto resvec: polymers) {
	void **ra;
	PyObject *r_array = python_voidp_array(resvec.size(), &ra);
	int i = 0;
        for (auto r: resvec)
	  ra[i++] = static_cast<void *>(r);
	PyTuple_SetItem(poly, p++, r_array);
  }	

  PyGILState_Release(gstate);
  return poly;
}

extern "C" void *molecule_new()
{
  AtomicStructure *m = new AtomicStructure();
  return m;
}

extern "C" void molecule_delete(void *mol)
{
  AtomicStructure *m = static_cast<AtomicStructure *>(mol);
  delete m;
}

extern "C" void *molecule_new_atom(void *mol, const char *atom_name, const char *element_name)
{
  AtomicStructure *m = static_cast<AtomicStructure *>(mol);
  Atom *a = m->new_atom(atom_name, Element(element_name));
  return a;
}

extern "C" void *molecule_new_bond(void *mol, void *atom1, void *atom2)
{
  AtomicStructure *m = static_cast<AtomicStructure *>(mol);
  Bond *b = m->new_bond(static_cast<Atom *>(atom1), static_cast<Atom *>(atom2));
  return b;
}

extern "C" void *molecule_new_residue(void *mol, const char *residue_name, const char *chain_id, int pos)
{
  AtomicStructure *m = static_cast<AtomicStructure *>(mol);
  Residue *r = m->new_residue(residue_name, chain_id, pos, ' ');
  return r;
}

static void *init_numpy()
{
  import_array(); // Initialize use of numpy
  return NULL;
}

// ---------------------------------------------------------------------------
// When a C++ object is deleted eliminate it from numpy arrays of pointers.
//
class Array_Updater : DestructionObserver
{
public:
  Array_Updater()
    {
      // Make sure we have the Python global interpreter lock.
      PyGILState_STATE gstate;
      gstate = PyGILState_Ensure();
      init_numpy();
      PyGILState_Release(gstate);
    }
  void add_array(PyObject *numpy_array)
    { arrays.insert(reinterpret_cast<PyArrayObject *>(numpy_array)); }
  void remove_array(void *numpy_array)
    { arrays.erase(reinterpret_cast<PyArrayObject *>(numpy_array)); }
  size_t array_count()
    { return arrays.size(); }
private:
  virtual void  destructors_done(const std::set<void*>& destroyed)
  {
    // Make sure we have the Python global interpreter lock.
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    for (auto a: arrays)
      filter_array(a, destroyed);

    PyGILState_Release(gstate);
  }
  void filter_array(PyArrayObject *a, const std::set<void*>& destroyed)
  {
    // Remove any destroyed pointers from numpy array and shrink the array in place.
    // Numpy array must be contiguous, 1 dimensional array.
    void **ae = static_cast<void **>(PyArray_DATA(a));
    npy_intp s = PyArray_SIZE(a);
    npy_intp j = 0;
    for (npy_intp i = 0; i < s ; ++i)
      if (destroyed.find(ae[i]) == destroyed.end())
	ae[j++] = ae[i];
    if (j < s)
      {
	//std::cerr << "resizing array " << a << " from " << s << " to " << j << std::endl;
	*PyArray_DIMS(a) = j;	// TODO: This hack may break numpy.
	/*
	// Numpy array can't be resized with weakref made by weakref.finalize().  Not sure why.
	// Won't work anyways because array will reallocate while looping over old array of atoms being deleted.
	PyArray_Dims dims;
	dims.len = 1;
	dims.ptr = &j;
	std::cerr << " base " << PyArray_BASE(a) << " weak " << ((PyArrayObject_fields *)a)->weakreflist << std::endl;
	if (PyArray_Resize(a, &dims, 0, NPY_CORDER) == NULL)
	  {
	    std::cerr << "Failed to delete molecule object pointers from numpy array." << std::endl;
	    PyErr_Print();
	  }
	*/
      }
  }
  std::set<PyArrayObject *> arrays;
};
class Array_Updater *array_updater = NULL;

extern "C" void remove_deleted_c_pointers(PyObject *numpy_array)
{
  if (array_updater == NULL)
    array_updater = new Array_Updater();

  array_updater->add_array(numpy_array);
}

extern "C" void pointer_array_freed(void *numpy_array)
{
  if (array_updater)
    {
      array_updater->remove_array(numpy_array);
      if (array_updater->array_count() == 0)
	{
	  delete array_updater;
	  array_updater = NULL;
	}
    }
}

class Object_Map_Deletion_Handler : DestructionObserver
{
public:
  Object_Map_Deletion_Handler(PyObject *object_map) : object_map(object_map) {}
private:
  PyObject *object_map;	// Dictionary from C++ pointer to Python wrapped object having a _c_pointer attribute.
  virtual void  destructors_done(const std::set<void*>& destroyed)
  {
    // Make sure we have the Python global interpreter lock.
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    remove_deleted_objects(destroyed);
    PyGILState_Release(gstate);
  }
  void remove_deleted_objects(const std::set<void*>& destroyed)
  {
    if (PyDict_Size(object_map) == 0)
      return;
    // TODO: Optimize by looping over object_map if it is smaller than destroyed.
    for (auto d: destroyed)
      {
	PyObject *dp = PyLong_FromVoidPtr(d);
	if (PyDict_Contains(object_map, dp))
	  {
	    PyObject *po = PyDict_GetItem(object_map, dp);
	    PyObject_DelAttrString(po, "_c_pointer");
	    PyObject_DelAttrString(po, "_c_pointer_ref");
	    PyDict_DelItem(object_map, dp);
	  }
	Py_DECREF(dp);
      }
  }
};

extern "C" void *object_map_deletion_handler(void *object_map)
{
  return new Object_Map_Deletion_Handler(static_cast<PyObject *>(object_map));
}

extern "C" void delete_object_map_deletion_handler(void *handler)
{
  delete static_cast<Object_Map_Deletion_Handler *>(handler);
}

