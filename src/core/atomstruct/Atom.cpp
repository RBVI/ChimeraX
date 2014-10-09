// vim: set expandtab ts=4 sw=4:
#include "Atom.h"
#include "AtomicStructure.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Residue.h"

#include <utility>  // for std::pair
#include <stdexcept>
#include <sstream>

namespace atomstruct {

Atom::Atom(AtomicStructure *as, const std::string &name, Element e):
    _name(name), _structure(as), _residue(NULL), _element(e),
    _coord_index(COORD_UNASSIGNED), _alt_loc(' '), _serial_number(-1),
    _aniso_u(NULL), BaseSphere<Bond, Atom>()
{
}

std::set<char>
Atom::alt_locs() const
{
    std::set<char> alt_locs;
    for (auto almi = _alt_loc_map.begin(); almi != _alt_loc_map.end();
    ++almi) {
        alt_locs.insert((*almi).first);
    }
    return alt_locs;
}

float
Atom::bfactor() const
{
    if (_alt_loc != ' ') {
        _Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
        return (*i).second.bfactor;
    }
    return structure()->active_coord_set()->get_bfactor(this);
}

void
Atom::_coordset_set_coord(const Point &coord)
{
    CoordSet *cs = structure()->active_coord_set();
    if (cs == NULL) {
        cs = structure()->find_coord_set(0);
        if (cs == NULL)
            cs = structure()->new_coord_set(0);
        structure()->set_active_coord_set(cs);
    }
    set_coord(coord, cs);
}

void
Atom::_coordset_set_coord(const Point &coord, CoordSet *cs)
{
    if (structure()->active_coord_set() == NULL)
        structure()->set_active_coord_set(cs);
    if (_coord_index == COORD_UNASSIGNED)
        _coord_index = _new_coord(coord);
    else if (_coord_index >= cs->coords().size()) {
        if (_coord_index > cs->coords().size()) {
            CoordSet *fill_cs = cs;
            while (fill_cs != NULL) {
                while (_coord_index > fill_cs->coords().size()) {
                    fill_cs->add_coord(Point());
                }
                fill_cs = structure()->find_coord_set(fill_cs->id()-1);
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
    auto& css = structure()->coord_sets();
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

const basegeom::Coord &
Atom::coord() const
{
    if (_coord_index == COORD_UNASSIGNED)
        throw std::logic_error("coordinate value hasn't been assigned");
    if (_alt_loc != ' ') {
        _Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
        return (*i).second.coord;
    }
    CoordSet *cs = structure()->active_coord_set();
    if (cs == NULL)
        throw std::logic_error("no active coordinate set");
    return cs->coords()[_coord_index];
}

Atom::IdatmInfoMap _idatm_map = {
	{ "Car", { Atom::Planar, 3, "aromatic carbon" } },
	{ "C3", { Atom::Tetrahedral, 4, "sp3-hybridized carbon" } },
	{ "C2", { Atom::Planar, 3, "sp2-hybridized carbon" } },
	{ "C1", { Atom::Linear, 2, "sp-hybridized carbon bonded to 2 other atoms" } },
	{ "C1-", { Atom::Linear, 1, "sp-hybridized carbon bonded to 1 other atom" } },
	{ "Cac", { Atom::Planar, 3, "carboxylate carbon" } },
	{ "N3+", { Atom::Tetrahedral, 4, "sp3-hybridized nitrogen, formal positive charge" } },
	{ "N3", { Atom::Tetrahedral, 3, "sp3-hybridized nitrogen, neutral" } },
	{ "Npl", { Atom::Planar, 3, "sp2-hybridized nitrogen, not double bonded" } },
	{ "N2+", { Atom::Planar, 3, "sp2-hybridized nitrogen, double bonded, formal positive charge" } },
	{ "N2", { Atom::Planar, 2, "sp2-hybridized nitrogen, double bonded" } },
	{ "N1+", { Atom::Linear, 2, "sp-hybridized nitrogen bonded to 2 other atoms" } },
	{ "N1", { Atom::Linear, 1, "sp-hybridized nitrogen bonded to 1 other atom" } },
	{ "Ntr", { Atom::Planar, 3, "nitro nitrogen" } },
	{ "Ng+", { Atom::Planar, 3, "guanidinium/amidinium nitrogen, partial positive charge" } },
	{ "O3", { Atom::Tetrahedral, 2, "sp3-hybridized oxygen" } },
	{ "O3-", { Atom::Tetrahedral, 1, "phosphate or sulfate oxygen sharing formal negative charge" } },
	{ "Oar", { Atom::Planar, 2, "aromatic oxygen" } },
	{ "Oar+", { Atom::Planar, 2, "aromatic oxygen, formal positive charge" } },
	{ "O2", { Atom::Planar, 1, "sp2-hybridized oxygen" } },
	{ "O2-", { Atom::Planar, 1, "carboxylate oxygen sharing formal negative charge; nitro group oxygen" } },
	{ "O1", { Atom::Linear, 1, "sp-hybridized oxygen" } },
	{ "O1+", { Atom::Linear, 1, "sp-hybridized oxygen, formal positive charge" } },
	{ "S3+", { Atom::Tetrahedral, 3, "sp3-hybridized sulfur, formal positive charge" } },
	{ "S3", { Atom::Tetrahedral, 2, "sp3-hybridized sulfur, neutral" } },
	{ "S3-", { Atom::Tetrahedral, 1, "thiophosphate sulfur, sharing formal negative charge" } },
	{ "S2", { Atom::Planar, 1, "sp2-hybridized sulfur" } },
	{ "Sar", { Atom::Planar, 2, "aromatic sulfur" } },
	{ "Sac", { Atom::Tetrahedral, 4, "sulfate, sulfonate, or sulfamate sulfur" } },
	{ "Son", { Atom::Tetrahedral, 4, "sulfone sulfur" } },
	{ "Sxd", { Atom::Tetrahedral, 3, "sulfoxide sulfur" } },
	{ "Pac", { Atom::Tetrahedral, 4, "phosphate phosphorus" } },
	{ "Pox", { Atom::Tetrahedral, 4, "P-oxide phosphorus" } },
	{ "P3+", { Atom::Tetrahedral, 4, "sp3-hybridized phosphorus, formal positive charge" } },
	{ "HC", { Atom::Single, 1, "hydrogen bonded to carbon" } },
	{ "H", { Atom::Single, 1, "other hydrogen" } },
	{ "DC", { Atom::Single, 1, "deuterium bonded to carbon" } },
	{ "D", { Atom::Single, 1, "other deuterium" } }
};

const Atom::IdatmInfoMap&
Atom::get_idatm_info_map()
{
    return _idatm_map;
}

float
Atom::occupancy() const
{
    if (_alt_loc != ' ') {
        _Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
        return (*i).second.occupancy;
    }
    return structure()->active_coord_set()->get_occupancy(this);
}

const Atom::Rings&
Atom::rings(bool cross_residues, int all_size_threshold,
        std::set<const Residue*>* ignore) const
{
    structure()->rings(cross_residues, all_size_threshold, ignore);
    return _rings;
}

void
Atom::set_alt_loc(char alt_loc, bool create, bool from_residue)
{
    if (alt_loc == _alt_loc || alt_loc == ' ')
        return;
    if (create) {
        if (_alt_loc_map.find(alt_loc) != _alt_loc_map.end()) {
            set_alt_loc(alt_loc, create=false);
            return;
        }
        _alt_loc_map.insert(std::pair<char, _Alt_loc_info>(alt_loc, _Alt_loc_info()));
        _alt_loc = alt_loc;
        return;
    }

    _Alt_loc_map::iterator i = _alt_loc_map.find(alt_loc);
    if (i == _alt_loc_map.end()) {
        std::stringstream msg;
        msg << "set_alt_loc(): atom " << name() << " in residue "
            << residue()->str() << " does not have an alt loc '"
            << alt_loc << "'";
        throw std::invalid_argument(msg.str().c_str());
    }
    if (from_residue) {
        _Alt_loc_info &info = (*i).second;
        _aniso_u = info.aniso_u;
        _coordset_set_coord(info.coord);
        _serial_number = info.serial_number;
        _alt_loc = alt_loc;
    } else {
        residue()->set_alt_loc(alt_loc);
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
        structure()->active_coord_set()->set_bfactor(this, bfactor);
}

void
Atom::set_coord(const basegeom::Coord &coord, CoordSet *cs)
{
    if (cs == NULL) {
        cs = structure()->active_coord_set();
        if (cs == NULL) {
            cs = structure()->find_coord_set(0);
            if (cs == NULL) {
                cs = structure()->new_coord_set(0);
            }
            structure()->set_active_coord_set(cs);
        }
    }
    
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        (*i).second.coord = coord;
        if (_coord_index == COORD_UNASSIGNED)
            _coord_index = _new_coord(coord);
    } else
        _coordset_set_coord(coord, cs);
}

void
Atom::set_occupancy(float occupancy)
{
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        (*i).second.occupancy = occupancy;
    } else
        structure()->active_coord_set()->set_occupancy(this, occupancy);
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

}  // namespace atomstruct
