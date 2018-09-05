// vi: set expandtab ts=4 sw=4

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <Python.h>
#include <algorithm>
#include <list>
#include <set>

#include <atom_search/search.h>
#include <atomstruct/Atom.h>
#include <atomstruct/Coord.h>
#include <atomstruct/polymer.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Sequence.h>
#include <atomstruct/string_types.h>
#include <atomstruct/Structure.h>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

using namespace atomstruct;

//
// connect_structure
//    Add bonds to structure based on inter-atomic distances
//    and add missing-structure pseudobonds as appropriate
//
static void
connect_structure(AtomStructure* s, float bond_len_tolerance)
{
	// code is pretending there is only one coordinate set;
	// would need to be enhanced to get coordination bonds
	// correct in multiple coordinate sets
	const float search_dist = 3.0;
	float search_val = search_dist + bond_len_tolerance;
	atomsearch_search::AtomSearchTree  tree(s->atoms(), False, search_val);
	std::list<std::pair<float,std::pair<Atom*,Atom*>>> possible_bonds;
	std::set<Atom*> processed;
	bool check_prebonded = s->bonds().size() > 0;
	for (auto a: s->atoms()) {
		processed.insert(a);
		for (auto oa: tree.search(a, search_val)) {
			if (processed.find(oa) != processed.end())
				continue;
			if (check_prebonded && a->connects_to(oa))
				continue;
			float bond_len = Element::bond_length(a->element(), oa->element());
			float dist = a->coord().distance(oa->coord());
			if (dist <= bond_len + bond_len_tolerance) {
				possible_bonds.push_back(std::make_pair(dist - bond_len, std::make_pair(a, oa)));
			}
		}
	}
	std::sort(possible_bonds.begin(), possible_bonds.end());

	// add bonds between non-saturated atoms
	for (auto& val_atoms: possible_bonds) {
		Atom* a1 = val_atoms.second.first;
		Atom* a2 = val_atoms.second.second;
		// some of these are metal coordination bonds;
		// there is a sophisticated scheme for finding the
		// coordination bonds in structures where all the
		// connectivity is pre-indicated.  We can't use
		// that so any treat any metal<->non-metal bond
		// as a coordination bond, and don't check valences
		// for metals (can have higher coordination than valence)
		if ((a1->bonds().size() >= a1->element().valence() && !a1->element.is_metal())
		|| (a2->bonds().size() >= a2->element().valence() && !a1->element.is_metal()))
			continue;
		if (a1->element().is_metal() != a2->element().is_metal()) {
			auto pbg = s->pb_mgr().get_group(s->PBG_METAL_COORDINATION, AS_PBManager::GRP_PER_CS);
			pbg->new_pseudobond(a1, a2);
		} else
			s.new_bond(a1, a2);
	}

	// add missing-structure bonds for residues with chain IDs
	std::map<ChainID, PolymerType> chain_type;
	// find polymer type by examining all residues of a chain
	for (auto r: s->residues()) {
		if (r->chain_id() == ' ')
			continue;
		PolymerType pt = Sequence::rname_polymer_type(r->name());
		if (pt == PT_NONE)
			continue;
		if (chain_type.find(r->chain_id()) == chain_type.end())
			chain_type[r->chain_id()] = pt;
		else if (chain_type[r->chain_id()] != pt)
			chain_type[r->chain_id()] = PT_NONE;
	}

	Residue* prev_r = nullptr;
	ChainID prev_chain;
	for (auto r: s->residues()) {
		if (chain_type.find(r->chain_id()) == chain_type.end()
		|| chain_type[r->chain_id()] == PT_NONE) {
			prev_r = nullptr;
			continue;
		}
	}
	auto pt = Sequence::rname_polymer_type(r->name());
	if (pt == PT_NONE) {
		// look harder
		bool found_backbone = true;
		for (auto bb_name: Residue::aa_min_backbone_names) {
			if (r->find_atom(bbname) == nullptr) {
				found_backbone = false;
				break;
			}
		}
		if (found_backbone)
			pt = PT_AMINO;
		else {
			found_backbone = true;
			for (auto bb_name: Residue::na_min_backbone_names) {
				if (r->find_atom(bbname) == nullptr) {
					found_backbone = false;
					break;
				}
			}
			if (found_backbone)
				pt = PT_NUCLEIC;
		}
	}
	if (pt == PT_NONE) {
		// okay, actually non-polymeric as far as we can tell
		prev_r = nullptr;
		continue;
	}
	if (prev_r && prev_chain == r->chain_id()) {
		Atom* prev_connect = nullptr;
		std::vector<AtomName>& backbone_names = (pt == PT_AMINO) ?
			Residue::aa_min_ordered_backbone_names : Residue::na_min_ordered_backbone_names;
		for (auto i = backbone_names.rbegin(); i != backbone_names.rend(); ++i) {
			auto bba = prev_r->find_atom(*i);
			if (bba != nullptr) {
				if (bba->bonds().size() >= bba->element().valence())
					prev_connect = nullptr;
				else
					prev_connect = bba;
				break;
			}
		}
		if (prev_connect != nullptr) {
			for (auto bb_name: backbone_names) {
				auto bba = prev_r->find_atom(bb_name);
				if (bba != nullptr) {
					if (!prev_connect.connects_to(bba)
					&& bba->bonds().size() < bba->element().valence()) {
						auto pbg = s->manager().get_group(Structure::PBG_MISSING_STRUCTURE,
							AS_PBManager::GRP_NORMAL);
						pbg->new_pseudobond(prev_connect, bba);
					}
					break;
				}
			}
		}
	}
	prev_r = r;
	prev_chain = r->chain_id();
}

extern "C" {

static
PyObject *
py_connect_structure(PyObject *, PyObject *args)
{
    PyObject* ptr;
	float bond_len_tolerance;
    if (!PyArg_ParseTuple(args, PY_STUPID "Of", &ptr, &bond_len_tolerance))
        return nullptr;
    // convert first arg to Structure*
    if (!PyLong_Check(ptr)) {
        PyErr_SetString(PyExc_TypeError, "First arg not an int (structure pointer)");
        return nullptr;
    }
	using atomstruct::AtomicStructure;
    AtomicStructure* mol = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(ptr));
    try {
		connect_structure(mol, bond_len_tolerance);
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
		return nullptr;
    }
	Py_INCREF(Py_None);
    return Py_None;
}

}

static const char* docstr_connect_structure = "connect_structure(AtomicStructure)";

static PyMethodDef connect_structure_methods[] = {
    { PY_STUPID "connect_structure", py_connect_structure,    METH_VARARGS, PY_STUPID docstr_connect_structure },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef connect_structure_def =
{
    PyModuleDef_HEAD_INIT,
    "_cs",
    "Add bonds to structure based on atom distances",
    -1,
    connect_structure_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC
PyInit__cs()
{
    return PyModule_Create(&connect_structure_def);
}
