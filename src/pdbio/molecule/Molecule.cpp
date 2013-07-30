#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Element.h"
#include "Molecule.h"
#include "Residue.h"

#include <algorithm>  // for std::find
#include <stdexcept>

Molecule::Molecule():
	_active_coord_set(NULL), asterisks_translated(false), lower_case_chains(false),
	pdb_version(0)
{
}

void
Molecule::delete_bond(Bond *b)
{
	Bonds::iterator i = std::find(_bonds.begin(), _bonds.end(), b);
	if (i == _bonds.end())
		throw std::invalid_argument("delete_bond called for Bond not in Molecule");

	b->_atoms[0]->remove_bond(b);
	b->_atoms[1]->remove_bond(b);

	_bonds.erase(i);
	delete b;
}

CoordSet *
Molecule::find_coord_set(int id) const
{
	for (CoordSets::const_iterator csi = _coord_sets.begin(); csi != _coord_sets.end();
	++csi) {
		if ((*csi)->id() == id)
			return *csi;
	}

	return NULL;
}
Residue *
Molecule::find_residue(std::string &chain_id, int pos, char insert) const
{
	for (Residues::const_iterator ri = _residues.begin(); ri != _residues.end(); ++ri) {
		Residue *r = *ri;
		if (r->position() == pos && r->chain_id() == chain_id
		&& r->insertion_code() == insert)
			return r;
	}
	return NULL;
}

Residue *
Molecule::find_residue(std::string &chain_id, int pos, char insert, std::string &name) const
{
	for (Residues::const_iterator ri = _residues.begin(); ri != _residues.end(); ++ri) {
		Residue *r = *ri;
		if (r->position() == pos && r->name() == name && r->chain_id() == chain_id
		&& r->insertion_code() == insert)
			return r;
	}
	return NULL;
}

Atom *
Molecule::new_atom(std::string &name, Element e)
{
	Atom *  baby = new Atom(this, name, e);
	_atoms.push_back(baby);
	return baby;
}

Bond *
Molecule::new_bond(Atom *a1, Atom *a2)
{
	Bond *baby = new Bond(this, a1, a2);
	_bonds.push_back(baby);
	return baby;
}

CoordSet *
Molecule::new_coord_set()
{
	if (_coord_sets.empty())
		return new_coord_set(0);
	return new_coord_set(_coord_sets.back()->id());
}

static void
_coord_set_insert(Molecule::CoordSets &coord_sets, CoordSet *cs, int index)
{
	if (coord_sets.empty() || coord_sets.back()->id() < index) {
		coord_sets.push_back(cs);
		return;
	}
	for (Molecule::CoordSets::iterator csi = coord_sets.begin(); csi != coord_sets.end();
	++csi) {
		CoordSet *cur_cs = *csi;
		if (index < cur_cs->id()) {
			coord_sets.insert(csi, cs);
			return;
		} else if (index == cur_cs->id()) {
			CoordSet *replaced = cur_cs;
			cur_cs = cs;
			delete replaced;
			return;
		}
	}
	std::logic_error("CoordSet insertion logic error");
}

CoordSet *
Molecule::new_coord_set(int index)
{
	if (!_coord_sets.empty())
		return new_coord_set(index, _coord_sets.back()->coords().size());
	CoordSet *cs = new CoordSet(index);
	_coord_set_insert(_coord_sets, cs, index);
	return cs;
}

CoordSet *
Molecule::new_coord_set(int index, int size)
{
	CoordSet *cs = new CoordSet(index, size);
	_coord_set_insert(_coord_sets, cs, index);
	return cs;
}

Residue *
Molecule::new_residue(std::string &name, std::string &chain, int pos, char insert,
	Residue *neighbor, bool after)
{
	Residue *baby = new Residue(this, name, chain, pos, insert);
	if (neighbor == NULL)
		_residues.push_back(baby);
	else {
		Residues::iterator ri = std::find(_residues.begin(), _residues.end(), neighbor);
		if (ri == _residues.end())
			throw std::out_of_range("Waypoint residue not in residue list");
		if (after)
			++ri;
		_residues.insert(ri, baby);
	}
	return baby;
}

void
Molecule::set_active_coord_set(CoordSet *cs)
{
	CoordSet *new_active;
	if (cs == NULL) {
		if (_coord_sets.empty())
			return;
		new_active = _coord_sets.front();
	} else {
		CoordSets::iterator csi = std::find(_coord_sets.begin(), _coord_sets.end(), cs);
		if (csi == _coord_sets.end())
			throw std::out_of_range("Requested active coord set not in coord sets");
		new_active = cs;
	}
	_active_coord_set = new_active;
}
