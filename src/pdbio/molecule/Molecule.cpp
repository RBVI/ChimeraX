#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Element.h"
#include "Molecule.h"
#include "Residue.h"

#include <algorithm>  // for std::find
#include <stdexcept>
#include <set>

Molecule::Molecule():
	_active_coord_set(NULL), asterisks_translated(false), lower_case_chains(false),
	pdb_version(0), is_traj(false)
{
}

std::map<Atom *, char>
Molecule::best_alt_locs() const
{
	// check the common case of all blank alt locs first...
	bool all_blank = true;
	for (Atoms::const_iterator ai = _atoms.begin(); ai != _atoms.end(); ++ai) {
		if (!(*ai)->_alt_loc_map.empty()) {
			all_blank = false;
			break;
		}
	}
	std::map<Atom *, char> best_locs;
	if (all_blank) {
		for (Atoms::const_iterator ai = _atoms.begin(); ai != _atoms.end(); ++ai) {
			best_locs.insert(std::pair<Atom *, char>(*ai, ' '));
		}
		return best_locs;
	}

	// non-blank alt locs present
	std::set<Atom *> seen;
	for (Atoms::const_iterator ai = _atoms.begin(); ai != _atoms.end(); ++ai) {
		Atom *a = *ai;
		if (seen.find(a) != seen.end())
			continue;
		seen.insert(a);
		std::set<Atom *> todo;
		todo.insert(a);
		std::set<char> alt_loc_set;
		if (!a->_alt_loc_map.empty()) {
			for (Atom::_Alt_loc_map::iterator ali = a->_alt_loc_map.begin();
					ali != a->_alt_loc_map.end(); ++ali) {
				alt_loc_set.insert((*ali).first);
			}
		}
		std::map<char, int> occurances;
		std::map<char, float> occupancies, bfactors;
		std::vector<Atom *> cur_atoms;
		while (!todo.empty()) {
			Atom *ta = *todo.begin();
			todo.erase(todo.begin());
			seen.insert(ta);
			cur_atoms.push_back(ta);
			if (!alt_loc_set.empty()) {
				for (Atom::_Alt_loc_map::iterator ali = a->_alt_loc_map.begin();
						ali != a->_alt_loc_map.end(); ++ali) {
					char alt_loc = (*ali).first;
					Atom::_Alt_loc_info info = (*ali).second;
					occurances[alt_loc] += 1;
					occupancies[alt_loc] += info.occupancy;
					bfactors[alt_loc] += info.bfactor;
				}
			}

			for (Atom::BondsMap::iterator bmi = a->_bonds.begin(); bmi != a->_bonds.end();
					++bmi) {
				Atom *nb = (*bmi).first;
				if (seen.find(nb) != seen.end())
					continue;
				if (alt_loc_set.empty()) {
					if (nb->_alt_loc_map.empty())
						todo.insert(nb);
				} else {
					for (std::set<char>::iterator ali = alt_loc_set.begin();
							ali != alt_loc_set.end(); ++ali) {
						if (nb->_alt_loc_map.find(*ali) != nb->_alt_loc_map.end()) {
							todo.insert(nb);
							break;
						}
					}
				}
			}
		}
		char best_loc;
		if (alt_loc_set.empty()) {
			best_loc = ' ';
		} else {
			int best_occurances = 0;
			float best_occupancies = 0.0, best_bfactors = 0.0;
			best_loc = '\0';
			for (std::set<char>::iterator ali = alt_loc_set.begin();
					ali != alt_loc_set.end(); ++ ali) {
				char alt_loc = *ali;
				bool is_best = best_loc == '\0';
				float occurance = occurances[alt_loc];
				if (!is_best) {
					if (occurance < best_occurances)
						continue;
					else if (occurance > best_occurances)
						is_best = true;
				}
				float occupancy = occupancies[alt_loc];
				if (!is_best) {
					if (occupancy < best_occupancies)
						continue;
					else if (occupancy > best_occupancies)
						is_best = true;
				}
				float bfactor = bfactors[alt_loc];
				if (!is_best) {
					if (bfactor > best_bfactors)
						continue;
					else if (bfactor < best_bfactors || alt_loc < best_loc)
						is_best = true;
				}
				if (is_best) {
					best_loc = alt_loc;
					best_occurances = occurance;
					best_occupancies = occupancy;
					best_bfactors = bfactor;
				}
			}
			for (std::vector<Atom *>::iterator ai = cur_atoms.begin(); ai != cur_atoms.end();
					++ai) {
				best_locs[*ai] = best_loc;
			}
		}
	}
	return best_locs;
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

void
Molecule::use_best_alt_locs()
{
	std::map<Atom *, char> alt_loc_map = best_alt_locs();
	for (std::map<Atom *, char>::iterator almi = alt_loc_map.begin();
			almi != alt_loc_map.end(); ++almi) {
		(*almi).first->set_alt_loc((*almi).second);
	}
}
