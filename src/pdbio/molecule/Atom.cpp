#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Molecule.h"
#include <utility>  // for std::pair
#include <stdexcept>

Atom::Atom(Molecule *m, std::string &name, Element e):
	_name(name), _molecule(m), _residue(NULL), _element(e),
	_coord_index(COORD_UNASSIGNED), _alt_loc(' '), _serial_number(-1),
	_aniso_u(NULL), BaseSphere<Bond, Atom>()
{
}

float
Atom::bfactor() const
{
	if (_alt_loc != ' ') {
		_Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
		return (*i).second.bfactor;
	}
	return molecule()->active_coord_set()->get_bfactor(this);
}

void
Atom::_coordset_set_coord(const Point &coord)
{
	CoordSet *cs = molecule()->active_coord_set();
	if (cs == NULL) {
		cs = molecule()->find_coord_set(0);
		if (cs == NULL)
			cs = molecule()->new_coord_set(0);
		molecule()->set_active_coord_set(cs);
	}
	set_coord(coord, cs);
}

void
Atom::_coordset_set_coord(const Point &coord, CoordSet *cs)
{
	if (molecule()->active_coord_set() == NULL)
		molecule()->set_active_coord_set(cs);
	if (_coord_index == COORD_UNASSIGNED)
		_coord_index = _new_coord(coord);
	else if (_coord_index >= cs->coords().size()) {
		if (_coord_index > cs->coords().size()) {
			CoordSet *fill_cs = cs;
			while (fill_cs != NULL) {
				while (_coord_index > fill_cs->coords().size()) {
					fill_cs->add_coord(Point());
				}
				fill_cs = molecule()->find_coord_set(fill_cs->id()-1);
			}
		}
		cs->add_coord(coord);
	} else {
		cs->_coords[_coord_index] = coord;
	}
}

unsigned int
Atom::_new_coord(const Point &coord)
{
	unsigned int index = COORD_UNASSIGNED;
	const Molecule::CoordSets& css = molecule()->coord_sets();
	for (auto csi = css.begin(); csi != css.end(); ++csi) {
		CoordSet *cs = (*csi).get();
		if (index == COORD_UNASSIGNED) {
			index = cs->coords().size();
			cs->add_coord(coord);
		} else
			while (index >= cs->coords().size())
				cs->add_coord(coord);
	}
	return index;
}

const Coord &
Atom::coord() const
{
	if (_coord_index == COORD_UNASSIGNED)
		throw std::logic_error("coordinate value hasn't been assigned");
	CoordSet *cs = molecule()->active_coord_set();
	if (cs == NULL)
		throw std::logic_error("no active coordinate set");
	return cs->coords()[_coord_index];
}

float
Atom::occupancy() const
{
	if (_alt_loc != ' ') {
		_Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
		return (*i).second.occupancy;
	}
	return molecule()->active_coord_set()->get_occupancy(this);
}

void
Atom::set_alt_loc(char alt_loc, bool create)
{
	if (alt_loc == _alt_loc || alt_loc == ' ')
		return;
	if (create) {
		if (_alt_loc_map.find(alt_loc) != _alt_loc_map.end()) {
			set_alt_loc(alt_loc, create=false);
			return;
		}
		_alt_loc_map.insert(std::pair<char, _Alt_loc_info>(alt_loc, _Alt_loc_info()));
		return;
	}

	_Alt_loc_map::iterator i = _alt_loc_map.find(alt_loc);
	if (i == _alt_loc_map.end())
		throw std::invalid_argument("Alternate location given to set_alt_loc()"
			" does not exist!");
	_Alt_loc_info &info = (*i).second;
	_aniso_u = info.aniso_u;
	_coordset_set_coord(info.coord);
	_serial_number = info.serial_number;
	_alt_loc = alt_loc;
	if (!create) {
		// set neighboring alt locs
		const BondsMap &bm = bonds_map();
		for (BondsMap::const_iterator bmi = bm.begin(); bmi != bm.end(); ++bmi) {
			Atom *nb = (*bmi).first;
			if (nb->_alt_loc_map.find(alt_loc) != nb->_alt_loc_map.end())
				nb->set_alt_loc(alt_loc);
		}
	}
}

void
Atom::set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33)
{
	if (_aniso_u == NULL) {
		_aniso_u = new std::vector<float>;
		_aniso_u->reserve(6);
		if (_alt_loc != ' ') {
			_Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
			(*i).second.aniso_u = _aniso_u;
		}
	}
	(*_aniso_u)[0] = u11;
	(*_aniso_u)[1] = u12;
	(*_aniso_u)[2] = u13;
	(*_aniso_u)[3] = u22;
	(*_aniso_u)[4] = u23;
	(*_aniso_u)[5] = u33;
}

void
Atom::set_bfactor(float bfactor)
{
	if (_alt_loc != ' ') {
		_Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
		(*i).second.bfactor = bfactor;
	} else
		molecule()->active_coord_set()->set_bfactor(this, bfactor);
}

void
Atom::set_coord(const Coord &coord, CoordSet *cs)
{
	if (cs == NULL) {
		cs = molecule()->active_coord_set();
		if (cs == NULL) {
			cs = molecule()->find_coord_set(0);
			if (cs == NULL) {
				cs = molecule()->new_coord_set(0);
			}
			molecule()->set_active_coord_set(cs);
		}
	}
	
	_coordset_set_coord(coord, cs);
}

void
Atom::set_occupancy(float occupancy)
{
	if (_alt_loc != ' ') {
		_Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
		(*i).second.occupancy = occupancy;
	} else
		molecule()->active_coord_set()->set_occupancy(this, occupancy);
}

void
Atom::set_serial_number(int sn)
{
	_serial_number = sn;
	if (_alt_loc != ' ') {
		_Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
		(*i).second.serial_number = sn;
	}
}
