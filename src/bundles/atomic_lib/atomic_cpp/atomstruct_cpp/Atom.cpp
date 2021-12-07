// vi: set expandtab ts=4 sw=4:

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

#include "Python.h"
#include <sstream>
#include <stdexcept>
#include <utility>  // for std::pair

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "Bond.h"
#include "ChangeTracker.h"
#include "CoordSet.h"
#include "destruct.h"
#include "Structure.h"
#include "PBGroup.h"
#include "Pseudobond.h"
#include "Residue.h"
#include "tmpl/residues.h"
#include "tmpl/TAexcept.h"

#include <pyinstance/PythonInstance.instantiate.h>
#include <pysupport/convert.h>

template class pyinstance::PythonInstance<atomstruct::Atom>;

namespace atomstruct {

Atom::Atom(Structure* as, const char* name, const Element& e):
    _alt_loc(' '), _aniso_u(nullptr), _coord_index(COORD_UNASSIGNED), _element(&e), _name(name),
    _radius(0.0), // -1 indicates not explicitly set
    _residue(nullptr), _ribbon_coord(nullptr), _serial_number(-1), _structure(as),
    _structure_category(Atom::StructCat::Unassigned)
{
    change_tracker()->add_created(structure(), this);
    structure()->_structure_cats_dirty = true;
}

Atom::~Atom()
{
    // Delete aniso vector for blank alt loc
    if (_aniso_u) {
        delete _aniso_u;
         _aniso_u = nullptr;
    }
    if (_ribbon_coord) {
      delete _ribbon_coord;
      _ribbon_coord = nullptr;
    }
    DestructionUser(this);
    if (selected()) {
        // so that closing a structure can fire "selection changed" trigger
        change_tracker()->add_modified(nullptr, this, ChangeTracker::REASON_SELECTED);
	graphics_changes()->set_gc_select(); // update ribbon selection highlight
    }

    change_tracker()->add_deleted(structure(), this);
    graphics_changes()->set_gc_adddel();
}

void
Atom::add_bond(Bond *b)
{
    _bonds.push_back(b);
    _neighbors.push_back(b->other_atom(this));
    graphics_changes()->set_gc_shape();
    _uncache_radius();
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

const std::vector<float> *
Atom::aniso_u() const
{
    if (_alt_loc != ' ') {
        _Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
        assert(i != _alt_loc_map.end());
        return (*i).second.aniso_u.get();
    }
    return _aniso_u;
}

const std::vector<float> *
Atom::aniso_u(char alt_loc) const
{
    if (alt_loc != ' ') {
        _Alt_loc_map::const_iterator i = _alt_loc_map.find(alt_loc);
        assert(i != _alt_loc_map.end());
        return (*i).second.aniso_u.get();
    }
    return _aniso_u;
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
Atom::clean_alt_locs()
{
    if (_alt_loc == ' ')
        return;
    auto alt_aniso_u = aniso_u();
    if (alt_aniso_u == nullptr) {
        if (_aniso_u != nullptr) {
            delete _aniso_u;
            _aniso_u = nullptr;
        }
    } else {
        if (_aniso_u == nullptr) {
            _aniso_u = new std::vector<float>(6);
        }
        (*_aniso_u)[0] = (*alt_aniso_u)[0];
        (*_aniso_u)[1] = (*alt_aniso_u)[1];
        (*_aniso_u)[2] = (*alt_aniso_u)[2];
        (*_aniso_u)[3] = (*alt_aniso_u)[3];
        (*_aniso_u)[4] = (*alt_aniso_u)[4];
        (*_aniso_u)[5] = (*alt_aniso_u)[5];
    }
    structure()->active_coord_set()->set_bfactor(this, bfactor());
    _coordset_set_coord(coord(), structure()->active_coord_set(), false);
    structure()->active_coord_set()->set_occupancy(this, occupancy());
    _serial_number = serial_number();

    _alt_loc = ' ';
    _alt_loc_map.clear();
}

void
Atom::clear_aniso_u()
{
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        assert(i != _alt_loc_map.end());
        if ((*i).second.aniso_u.use_count() > 0) {
            (*i).second.aniso_u.reset();
            change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_ANISO_U);
        }
    } else if (_aniso_u != nullptr) {
        delete _aniso_u;
        _aniso_u = nullptr;
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_ANISO_U);
    }
}

void
Atom::_coordset_set_coord(const Point &coord, CoordSet *cs, bool track_change)
{
    if (structure()->active_coord_set() == nullptr)
        structure()->set_active_coord_set(cs);
    if (_coord_index == COORD_UNASSIGNED)
        _coord_index = _new_coord(coord);
    else if (_coord_index >= cs->coords().size()) {
        if (_coord_index > cs->coords().size()) {
            CoordSet *fill_cs = cs;
            while (fill_cs != nullptr) {
                while (_coord_index > fill_cs->coords().size()) {
                    fill_cs->add_coord(Point());
                }
                fill_cs = structure()->find_coord_set(fill_cs->id()-1);
            }
        }
        cs->add_coord(coord);
        graphics_changes()->set_gc_shape();
        if (in_ribbon())
            graphics_changes()->set_gc_ribbon();
        graphics_changes()->set_gc_ring();
    } else {
        //cs->_coords[_coord_index] = coord;
        cs->_coords[_coord_index].set_xyz(coord[0], coord[1], coord[2]);
        if (track_change) {
            graphics_changes()->set_gc_shape();
            if (in_ribbon())
                graphics_changes()->set_gc_ribbon();
            graphics_changes()->set_gc_ring();
            change_tracker()->add_modified(structure(), cs, ChangeTracker::REASON_COORDSET);
            if (structure()->active_coord_set() == cs)
                structure()->change_tracker()->add_modified(structure(), structure(),
                    ChangeTracker::REASON_SCENE_COORD);
        }
    }
}

unsigned int
Atom::_new_coord(const Point &coord)
{
    unsigned int index = COORD_UNASSIGNED;
    auto& css = structure()->coord_sets();
    for (auto csi = css.begin(); csi != css.end(); ++csi) {
        CoordSet *cs = *csi;
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
    CoordSet *cs = structure()->active_coord_set();
    if (cs == nullptr)
        throw std::logic_error("no active coordinate set");
    return coord(cs);
}

const Coord&
Atom::coord(const CoordSet* cs) const
{
    if (_coord_index == COORD_UNASSIGNED)
        throw std::logic_error("coordinate value hasn't been assigned");
    if (_alt_loc != ' ') {
        _Alt_loc_map::const_iterator i = _alt_loc_map.find(_alt_loc);
        return (*i).second.coord;
    }
    return cs->coords()[_coord_index];
}

int
Atom::coordination(int value_if_unknown) const
{
    if (bonds().size() > 0)
        return bonds().size();
    int num_pb = 0;
    auto pbg = structure()->pb_mgr().get_group(Structure::PBG_METAL_COORDINATION);
    if (pbg != nullptr) {
        for (auto pb: pbg->pseudobonds()) {
            for (auto a: pb->atoms()) {
                if (a == this) {
                    ++num_pb;
                    break;
                }
            }
        }
    }
    return num_pb || value_if_unknown;
}

float
Atom::default_radius() const
{
    // for metal cations:
    // based on ionic radii from Handbook of Chemistry and Physics (2001),
    // assume cations are +2 (if no such state, then lowest existing)
    // for missing elements (e.g. holmium) used Ionic_radius entry from
    // Wikipedia ("Effective" ionic radius).
    //
    // for "united atom" organics:
    // based on "The Packing Density in Proteins: Standard Radii and Volumes"
    //    Tsai, Taylor, Chothia, and Gerstein, JMC 290, 253-266 (1999).
    //
    // for "explicit hydrogen" organics:
    // use Amber 99 FF values
    //
    // for others:
    // double the bond radius found in Allen et al,
    // Acta Cryst. Sect. B 35, 2331 (1979)

    static char Car[] = "Car";
    static char C2[] = "C2";
    static char O3[] = "O3";

    // for explicit hydrogens, use Bondi values
    if (structure()->num_hyds() > 0) {
        const IdatmInfoMap &type_info = get_idatm_info_map();
        auto info = type_info.find(idatm_type());
        if (info != type_info.end()
        && info->second.substituents <= bonds().size()) {
            switch (element().number()) {
                case 1: // hydrogen
                    return 1.00f;

                case 6: // carbon
                    return 1.70f;

                case 7: // nitrogen
                    return 1.625f;

                case 8: // oxygen
                    if (info->second.geometry < 4)
                        return 1.48f;
                    return 1.50f;

                case 9: // fluorine
                    return 1.56f;

                case 15: // phosphorus
                    return 1.871f;

                case 16: // sulfur
                    return 1.782f;

                case 17: // chlorine
                    return 1.735f;

                case 35: // bromine
                    return 1.978f;

                case 53: // iodine
                    return 2.094f;
            }
        }
    }

    // use JMB united atom values or ionic radii
    int coord;
    switch (element().number()) {
        case 1: // hydrogen
            return 1.0f;
        
        case 3: // lithium
            coord = coordination(6);
            if (coord <= 4)
                return 0.59f;
            if (coord >= 8)
                return 0.92f;
            return 0.76f;

        case 4: // beryllium
            coord = coordination(6);
            if (coord <= 4)
                return 0.27f;
            return 0.45f;

        case 6: // carbon
            if (idatm_type() != Car && idatm_type() != C2)
                return 1.88f;
            if (bonds().size() < 3) // implied hydrogen
                return 1.76f;
            if (structure()->num_hyds() > 0)
                for (auto nb: neighbors()) {
                    if (nb->element().number() == 1)
                        return 1.76f;
                }
            return 1.61f;
        
        case 7: // nitrogen
            return 1.64f;
        
        case 8: // oxygen
            if (idatm_type() == O3)
                return 1.46f;
            return 1.42f;
        
        case 9: // fluorine
            // if it's an ion, use the ionic value
            if (bonds().size() == 0)
                return 1.33f;
            break;

        case 11: // sodium
            coord = coordination(6);
            if (coord <= 4)
                return 0.99f;
            if (coord >= 12)
                return 1.39f;
            if (coord >= 9)
                return 1.24f;
            if (coord >= 8)
                return 1.18f;
            return 1.02f;
        
        case 12: // magnesium
            coord = coordination(6);
            if (coord <= 4)
                return 0.57f;
            if (coord >= 8)
                return 0.89f;
            return 0.72f;
        
        case 13: // aluminum
            coord = coordination(6);
            if (coord <= 4)
                return 0.39f;
            if (coord >= 6)
                return 0.54f;
            return 0.48f;
        
        case 15: // phosphorus
            // to be consistent with "explicit hydrogen" case
            return 2.1f;
        
        case 16: // sulfur
            return 1.77f;
        
        case 17: // chlorine
            if (bonds().size() == 0)
                return 1.81f;
            break;

        case 19: // potassium
            coord = coordination(6);
            if (coord <= 4)
                return 1.37f;
            if (coord >= 12)
                return 1.64;
            if (coord >= 10)
                return 1.51f;
            return 1.38f;
        
        case 20: // calcium
            coord = coordination(6);
            if (coord >= 12)
                return 1.34;
            if (coord >= 10)
                return 1.23f;
            if (coord >= 8)
                return 1.12f;
            return 1.00f;
        
        case 21: // scandium
            coord = coordination(6);
            if (coord >= 8)
                return 0.87f;
            return 0.75f;

        case 22: // titanium
            return 0.86f;

        case 23: // vanadium
            return 0.79f;

        case 24: // chromium
            return 0.73f;

        case 25: // manganese
            coord = coordination(6);
            if (coord <= 4)
                return 0.66f;
            if (coord >= 8)
                return 0.96f;
            return 0.83f;

        case 26: // iron
            coord = coordination(6);
            if (coord <= 4)
                return 0.63f;
            if (coord >= 8)
                return 0.92f;
            return 0.61f;

        case 27: // cobalt
            coord = coordination(6);
            if (coord <= 4)
                return 0.56f;
            if (coord >= 8)
                return 0.90f;
            return 0.65f;
            
        case 28: // nickel
            coord = coordination(6);
            if (coord <= 4)
                return 0.49f;
            return 0.69f;

        case 29: // copper
            coord = coordination(6);
            if (coord <= 4)
                return 0.57f;
            return 0.73f;
        
        case 30: // zinc
            coord = coordination(6);
            if (coord <= 4)
                return 0.60f;
            if (coord >= 8)
                return 0.90f;
            return 0.74f;

        case 31: // gallium
            coord = coordination(6);
            if (coord <= 4)
                return 0.47f;
            return 0.62f;

        case 32: // germanium
            return 0.73f;

        case 33: // arsenic
            return 0.58f;

        case 34: // selenium
            return 0.50f;

        case 35: // bromine
            if (bonds().size() == 0)
                return 1.96f;
            break;

        case 37: // rubidium
            coord = coordination(6);
            if (coord >= 12)
                return 1.72f;
            if (coord >= 10)
                return 1.66f;
            if (coord >= 8)
                return 1.61f;
            return 1.52f;
        
        case 38: // strontium
            coord = coordination(6);
            if (coord >= 12)
                return 1.44f;
            if (coord >= 10)
                return 1.36f;
            if (coord >= 8)
                return 1.26f;
            return 1.18f;
        
        case 39: // yttrium
            coord = coordination(6);
            if (coord >= 9)
                return 1.08f;
            if (coord >= 8)
                return 1.02f;
            return 0.90f;

        case 40: // zirconium
            coord = coordination(6);
            if (coord <= 4)
                return 0.59f;
            if (coord >= 9)
                return 0.89f;
            if (coord >= 8)
                return 0.84f;
            return 0.72f;

        case 41: // niobium
            coord = coordination(6);
            if (coord >= 8)
                return 0.79f;
            return 0.72f;

        case 42: // molybdenum
            return 0.69f;

        case 43: // technetium
            return 0.65f;

        case 44: // ruthenium
            return 0.68f;

        case 45: // rhodium
            return 0.67f;

        case 46: // palladium
            coord = coordination(6);
            if (coord <= 4)
                return 0.64f;
            return 0.86f;

        case 47: // silver
            coord = coordination(6);
            if (coord <= 4)
                return 0.79f;
            return 0.94f;

        case 48: // cadmium
            coord = coordination(6);
            if (coord <= 4)
                return 0.78f;
            if (coord >= 12)
                return 1.31f;
            if (coord >= 8)
                return 1.10f;
            return 0.95f;

        case 49: // indium
            coord = coordination(6);
            if (coord <= 4)
                return 0.62f;
            return 0.80f;

        case 50: // tin
            coord = coordination(6);
            if (coord <= 4)
                return 0.55f;
            if (coord >= 8)
                return 0.81f;
            return 0.69f;

        case 51: // antimony
            return 0.76f;

        case 52: // tellurium
            coord = coordination(6);
            if (coord <= 4)
                return 0.66f;
            return 0.97f;

        case 53: // iodine
            if (bonds().size() == 0)
                return 2.20f;
            break;

        case 55: // cesium
            coord = coordination(6);
            if (coord >= 12)
                return 1.88f;
            if (coord >= 10)
                return 1.81f;
            if (coord >= 8)
                return 1.74f;
            return 1.67f;

        case 56: // barium
            coord = coordination(6);
            if (coord >= 12)
                return 1.61f;
            if (coord >= 8)
                return 1.42f;
            return 1.35f;

        case 57: // lanthanum
            coord = coordination(6);
            if (coord >= 12)
                return 1.36f;
            if (coord >= 10)
                return 1.27f;
            if (coord >= 8)
                return 1.16f;
            return 1.03f;

        case 58: // cerium
            coord = coordination(6);
            if (coord >= 12)
                return 1.34f;
            if (coord >= 10)
                return 1.25f;
            if (coord >= 8)
                return 1.14f;
            return 1.01f;

        case 59: // praseodymium
            coord = coordination(6);
            if (coord >= 8)
                return 1.13f;
            return 0.99f;

        case 60: // neodymium
            coord = coordination(6);
            if (coord >= 12)
                return 1.27f;
            if (coord >= 9)
                return 1.16f;
            if (coord >= 8)
                return 1.12f;
            return 0.98f;

        case 61: // promethium
            coord = coordination(6);
            if (coord >= 8)
                return 1.09f;
            return 0.97f;

        case 62: // samarium
            coord = coordination(6);
            if (coord >= 8)
                return 1.27f;
            return 1.19f;
            
        case 63: // europium
            coord = coordination(6);
            if (coord >= 10)
                return 1.35f;
            if (coord >= 8)
                return 1.25f;
            return 1.17f;

        case 64: // gadolinium
            coord = coordination(6);
            if (coord >= 8)
                return 1.05f;
            return 0.94f;

        case 65: // terbium
            coord = coordination(6);
            if (coord >= 8)
                return 1.04f;
            return 0.92f;
            
        case 66: // dysprosium
            coord = coordination(6);
            if (coord >= 8)
                return 1.19f;
            return 1.07f;
            
        case 67: // holmium
            return 0.901f;

        case 68: // erbium
            coord = coordination(6);
            if (coord >= 8)
                return 1.00f;
            return 0.89f;
            
        case 69: // thulium
            coord = coordination(6);
            if (coord >= 7)
                return 1.09f;
            return 1.01f;
            
        case 70: // ytterbium
            coord = coordination(6);
            if (coord >= 8)
                return 1.14f;
            return 1.02f;
            
        case 71: // lutetium
            coord = coordination(6);
            if (coord >= 8)
                return 0.97f;
            return 0.86f;

        case 72: // hafnium
            coord = coordination(6);
            if (coord <= 4)
                return 0.58f;
            if (coord >= 8)
                return 0.83f;
            return 0.71f;

        case 73: // tantalum
            return 0.72f;

        case 74: // tungsten
            return 0.66f;

        case 75: // rhenium
            return 0.63f;

        case 76: // osmium
            return 0.63f;

        case 77: // iridium
            return 0.68f;

        case 78: // platinum
            coord = coordination(6);
            if (coord <= 4)
                return 0.60f;
            return 0.80f;

        case 79: // gold
            return 1.37f;

        case 80: // mercury
            coord = coordination(6);
            if (coord <= 2)
                return 0.69f;
            if (coord <= 4)
                return 0.96f;
            if (coord >= 8)
                return 1.14f;
            return 1.02f;

        case 81: // thallium
            coord = coordination(6);
            if (coord >= 12)
                return 1.70f;
            if (coord >= 8)
                return 1.59f;
            return 1.50f;

        case 82: // lead
            coord = coordination(6);
            if (coord >= 12)
                return 1.49f;
            if (coord >= 10)
                return 1.40f;
            if (coord >= 8)
                return 1.29f;
            return 1.19f;

        case 83: // bismuth
            coord = coordination(6);
            if (coord <= 5)
                return 0.96f;
            if (coord >= 8)
                return 1.17f;
            return 1.03f;

        case 84: // polonium
            return 0.97f;

        case 87: // francium
            return 1.80f;

        case 88: // radium
            coord = coordination(6);
            if (coord >= 10)
                return 1.70f;
            return 1.48f;

        case 89: // actinium
            return 1.12f;

        case 90: // thorium
            coord = coordination(6);
            if (coord >= 12)
                return 1.21f;
            if (coord >= 10)
                return 1.13f;
            if (coord >= 8)
                return 1.05f;
            return 0.94f;

        case 91: // protactinium
            return 1.04f;

        case 92: // uranium
            return 1.03f;
        
        case 93: // neptunium
            return 1.01f;

        case 94: // plutonium
            return 1.00f;

        case 95: // americium
            coord = coordination(6);
            if (coord >= 8)
                return 1.09f;
            return 0.98f;

        case 96: // curium
            return 0.97f;

        case 97: // berkelium
            return 0.96f;

        case 98: // californium
            return 0.95f;

        case 99: // einsteinium
            return 0.835f;
    }

    // use double the bond radius
    float rad = 2.0 * Element::bond_radius(element());
    if (rad == 0.0)
        return 1.8f;
    return rad;
}

void
Atom::delete_alt_loc(char al)
{
    // doesn't do any real consistency checking; use Residue::delete_alt_loc for that
    if (al == ' ')
        throw std::invalid_argument("Atom.delete_alt_loc(): cannot delete the ' ' alt loc");
    auto al_i = _alt_loc_map.find(al);
    if (al_i == _alt_loc_map.end()) {
        std::stringstream msg;
        msg << "delete_alt_loc(): atom " << name() << " in residue "
            << residue()->str() << " does not have an alt loc '" << al << "'";
        throw std::invalid_argument(msg.str().c_str());
    }
    _alt_loc_map.erase(al_i);
    if (al == _alt_loc) {
        if (_alt_loc_map.empty()) {
            _alt_loc = ' ';
        } else {
            _alt_loc = (*_alt_loc_map.begin()).first;
        }
        if (structure()->alt_loc_change_notify())
            graphics_changes()->set_gc_shape();
    }
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

bool
Atom::has_missing_structure_pseudobond() const
{
    auto pbg = structure()->pb_mgr().get_group(
        Structure::PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg != nullptr) {
        for (auto& pb: pbg->pseudobonds()) {
            if (pb->atoms()[0] == this || pb->atoms()[1] == this)
                return true;
        }
    }
    return false;
}

bool
Atom::is_backbone(BackboneExtent bbe) const
{
    // hydrogens depend on the heavy atom they're attached to, but are never in minimal backbone
    if (element().number() == 1) {
        if (bbe == BBE_MIN)
            return false;
        if (bonds().size() == 1) {
            auto bonded = *neighbors().begin();
            // need to check neighbor element to prevent possible infinite loop for H2
            return bonded->element().number() > 1 && bonded->is_backbone(bbe);
        }
        return false;
    }
    const std::set<AtomName>* bb_names = residue()->backbone_atom_names(bbe);
    if (bb_names == nullptr)
        return false;
    return bb_names->find(name()) != bb_names->end();
}

bool
Atom::is_ribose() const {
    // hydrogens depend on the heavy atom they're attached to
    if (element().number() == 1) {
        if (bonds().size() == 1) {
            auto bonded = *neighbors().begin();
            // need to check neighbor element to prevent possible infinite loop for H2
            return bonded->element().number() > 1 && bonded->is_ribose();
        }
        return false;
    }
    const std::set<AtomName>* rb_names = residue()->ribose_atom_names();
    if (rb_names == nullptr)
        return false;
    return rb_names->find(name()) != rb_names->end();
}

bool
Atom::is_side_connector() const
{
    // hydrogens depend on the heavy atom they're attached to
    if (element().number() == 1) {
        if (bonds().size() == 1) {
            auto bonded = *neighbors().begin();
            // need to check neighbor element to prevent possible infinite loop for H2
            return bonded->element().number() > 1 && bonded->is_side_connector();
        }
        return false;
    }
    const std::set<AtomName>* sc_names = residue()->side_connector_atom_names();
    if (sc_names == nullptr)
        return false;
    return sc_names->find(name()) != sc_names->end();
}

bool
Atom::is_side_chain(bool only) const {
    // hydrogens depend on the heavy atom they're attached to
    if (element().number() == 1) {
        if (bonds().size() == 1) {
            auto bonded = *neighbors().begin();
            // need to check neighbor element to prevent possible infinite loop for H2
            return bonded->element().number() > 1 && bonded->is_side_chain(only);
        }
        return false;
    }
    const std::set<AtomName>* bb_names = residue()->backbone_atom_names(BBE_MAX);
    if (bb_names == nullptr)
        return false;
    if (!only && is_side_connector())
        return true;
    return !is_backbone(BBE_MAX);
}

float
Atom::maximum_bond_radius(float default_radius) const {
    int bcount = 0;
    float rmax = 0.0;
    for (auto &b : bonds()) {
        float r = b->radius();
    if (r > rmax)
        rmax = r;
    bcount += 1;
    }
    return (bcount > 0 ? rmax : default_radius);
}

bool
Atom::is_missing_heavy_template_neighbors(bool chain_start, bool chain_end, bool no_template_okay) const
{
    auto tmpl_res = tmpl::find_template_residue(residue()->name(), chain_start, chain_end);
    if (tmpl_res == nullptr) {
        if (no_template_okay)
            return false;
        std::ostringstream os;
        os << "No residue template found for " << residue()->name();
        throw tmpl::TA_NoTemplate(os.str());
    }
    auto nm_ptr_i = tmpl_res->atoms_map().find(name());
    if (nm_ptr_i == tmpl_res->atoms_map().end())
        return false;
    auto tmpl_atom = nm_ptr_i->second;
    // non-standard input may not match the template atom names; just count bonded heavies
    int heavys = 0, tmpl_heavys = 0;
    for (auto nb: neighbors())
        if (nb->element().number() > 1 && nb->residue() == residue())
            ++heavys;
    for (auto tnb: tmpl_atom->neighbors())
        if (tnb->element().number() > 1
        && !(tnb->element().number() == 15 && tmpl_atom->name() != "OP1" && tmpl_atom->name() != "OP2"))
            // okay for nucleic phosphorus to be missing (but not to OP1/2!)
            ++tmpl_heavys;
    return heavys < tmpl_heavys;
}

Atom::Bonds::size_type
Atom::num_explicit_bonds() const // includes missing-structure bonds
{
    auto count = bonds().size();
    auto pbg = structure()->pb_mgr().get_group(
        Structure::PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg != nullptr) {
        for (auto& pb: pbg->pseudobonds()) {
            if (pb->atoms()[0] == this)
                ++count;
            if (pb->atoms()[1] == this)
                ++count;
        }
    }
    return count;
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

void
Atom::remove_bond(Bond *b)
{
    auto bi = std::find(_bonds.begin(), _bonds.end(), b);
    _neighbors.erase(_neighbors.begin() + (bi - _bonds.begin()));
    _bonds.erase(bi);
    graphics_changes()->set_gc_shape();
    _uncache_radius();
}

const Atom::Rings&
Atom::rings(bool cross_residues, int all_size_threshold,
        std::set<const Residue*>* ignore) const
{
    structure()->rings(cross_residues, all_size_threshold, ignore);
    return _rings;
}

Coord
Atom::effective_coord() const
{
    if (_residue && _residue->ribbon_display() && !visible()) {
        const Coord *c = ribbon_coord();
        if (c != nullptr)
            return *c;
    }
    return coord();
}

Coord
Atom::scene_coord() const
{
    return coord().mat_mul(structure()->position());

}

Coord
Atom::effective_scene_coord() const
{
    return effective_coord().mat_mul(structure()->position());

}

Coord
Atom::effective_scene_coord() const
{
    return mat_mul(structure()->position(), effective_coord());

}

Coord
Atom::scene_coord(const CoordSet* cs) const
{
    return coord(cs).mat_mul(structure()->position());
}

Coord
Atom::scene_coord(char alt_loc) const
{
    return coord(alt_loc).mat_mul(structure()->position());
}

int
Atom::session_num_floats(int version) const
{
    int num_floats = SESSION_NUM_FLOATS(version) + Rgba::session_num_floats()
        + _alt_loc_map.size() * SESSION_ALTLOC_FLOATS(version) +
        (_aniso_u == nullptr ? 0 : _aniso_u->size());
    for (auto &altloc_info : _alt_loc_map) {
        auto& info = altloc_info.second;
        if (info.aniso_u != nullptr)
            num_floats += info.aniso_u->size();
    }
    return num_floats;
}

void
Atom::session_restore(int version, int** ints, float** floats, PyObject* misc)
{
    using pysupport::pystring_to_cchar;

    _rgba.session_restore(ints, floats);
    auto& int_ptr = *ints;
    _alt_loc = int_ptr[0];
    auto num_alt_locs = int_ptr[1];
    auto aniso_u_size = int_ptr[2];
    _coord_index = int_ptr[3];
    _serial_number = int_ptr[4];
    _structure_category = (StructCat)int_ptr[5];
    _draw_mode = (DrawMode)int_ptr[6];
    _display = int_ptr[7];
    _hide = int_ptr[8];
    _selected = int_ptr[9];
    int_ptr += SESSION_NUM_INTS(version);

    auto& float_ptr = *floats;
    _radius = float_ptr[0];
    float_ptr += SESSION_NUM_FLOATS(version);

    if (!(PyTuple_Check(misc) || PyList_Check(misc)))
        throw std::invalid_argument("Atom misc info is not a tuple");
    if (PySequence_Fast_GET_SIZE(misc) != 2)
        throw std::invalid_argument("Atom misc info wrong size");
    _computed_idatm_type = pystring_to_cchar(PySequence_Fast_GET_ITEM(misc, 0), "computed IDATM type");
    _explicit_idatm_type = pystring_to_cchar(PySequence_Fast_GET_ITEM(misc, 1), "explicit IDATM type");

    if (aniso_u_size > 0) {
        _aniso_u = new std::vector<float>();
        _aniso_u->reserve(aniso_u_size);
        for (decltype(aniso_u_size) i = 0; i < aniso_u_size; ++i)
            _aniso_u->push_back(*float_ptr++);
    } else {
        _aniso_u = nullptr;
    }

    _alt_loc_map.clear();
    for (decltype(num_alt_locs) i = 0; i < num_alt_locs; ++i) {
        _Alt_loc_map::key_type alt_loc = int_ptr[0];
        aniso_u_size = int_ptr[1];
        _Alt_loc_info info;
        info.serial_number = int_ptr[2];
        int_ptr += SESSION_ALTLOC_INTS(version);

        info.bfactor = float_ptr[0];
        info.coord[0] = float_ptr[1];
        info.coord[1] = float_ptr[2];
        info.coord[2] = float_ptr[3];
        info.occupancy = float_ptr[4];
        float_ptr += SESSION_ALTLOC_FLOATS(version);

        if (aniso_u_size > 0) {
            info.aniso_u = std::make_shared<std::vector<float>>();
            info.aniso_u.get()->reserve(aniso_u_size);
            for (decltype(aniso_u_size) j = 0; j < aniso_u_size; ++j)
                info.aniso_u->push_back(*float_ptr++);
        } else {
            info.aniso_u = nullptr;
        }
        _alt_loc_map[alt_loc] = info;
    }
}

void
Atom::session_save(int** ints, float** floats, PyObject* misc) const
{
    color().session_save(ints, floats);
    auto& int_ptr = *ints;
    int_ptr[0] = _alt_loc;
    int_ptr[1] = _alt_loc_map.size();
    int_ptr[2] = (_aniso_u == nullptr ? 0 : _aniso_u->size());
    int_ptr[3] = _coord_index;
    int_ptr[4] = _serial_number;
    int_ptr[5] = (int)_structure_category;
    int_ptr[6] = (int)_draw_mode;
    int_ptr[7] = (int)_display;
    int_ptr[8] = (int)_hide;
    int_ptr[9] = (int)_selected;
    int_ptr += SESSION_NUM_INTS();

    auto& float_ptr = *floats;
    float_ptr[0] = _radius;
    float_ptr += SESSION_NUM_FLOATS();

    using pysupport::cchar_to_pystring;
    if (PyList_Append(misc, cchar_to_pystring(_computed_idatm_type, "computed IDATM type")) < 0)
        throw std::runtime_error("Cannot append computed atom type to misc list");
    if (PyList_Append(misc, cchar_to_pystring(_explicit_idatm_type, "explicit IDATM type")) < 0)
        throw std::runtime_error("Cannot append explicit atom type to misc list");

    if (_aniso_u != nullptr) {
        for (auto v: *_aniso_u) {
            *float_ptr = v;
            float_ptr++;
        }
    }

    for (auto &altloc_info : _alt_loc_map) {
        auto alt_loc = altloc_info.first;
        auto& info = altloc_info.second;
        int_ptr[0] = alt_loc;
        int_ptr[1] = (info.aniso_u == nullptr ? 0 : info.aniso_u->size());
        int_ptr[2] = info.serial_number;
        int_ptr += SESSION_ALTLOC_INTS();

        float_ptr[0] = info.bfactor;
        float_ptr[1] = info.coord[0];
        float_ptr[2] = info.coord[1];
        float_ptr[3] = info.coord[2];
        float_ptr[4] = info.occupancy;
        float_ptr += SESSION_ALTLOC_FLOATS();

        if (info.aniso_u != nullptr) {
            for (auto v: *info.aniso_u) {
                *float_ptr = v;
                float_ptr++;
            }
        }
    }
}

void
Atom::set_alt_loc(char alt_loc, bool create, bool _from_residue)
{
    if (alt_loc == _alt_loc || alt_loc == ' ')
        return;
    if (structure()->alt_loc_change_notify()) {
        graphics_changes()->set_gc_shape();
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_ALT_LOC);
    }
    if (create) {
        if (_alt_loc_map.find(alt_loc) != _alt_loc_map.end()) {
            set_alt_loc(alt_loc, create=false);
            return;
        }
        Coord crd;
        float bf;
        int sn;
        if (_coord_index != COORD_UNASSIGNED) {
            crd = coord();
            bf = bfactor();
            sn = serial_number();
        }
        _alt_loc_map[alt_loc];    // Create map entry.
        _alt_loc = alt_loc;
        if (_coord_index != COORD_UNASSIGNED) {
            set_coord(crd);
            set_bfactor(bf);
            set_serial_number(sn);
        }
        set_occupancy(0.0);
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
    if (_from_residue) {
        _Alt_loc_info &info = (*i).second;
        _serial_number = info.serial_number;
        _alt_loc = alt_loc;
        if (structure()->alt_loc_change_notify()) {
            change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_COORD);
            structure()->change_tracker()->add_modified(structure(), structure(),
                ChangeTracker::REASON_SCENE_COORD);
        }
    } else {
        residue()->set_alt_loc(alt_loc);
    }
}

void
Atom::set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33)
{
    std::vector<float> *a;
    if (_alt_loc == ' ') {
        if (_aniso_u == nullptr)
            _aniso_u = new std::vector<float>(6);
        a = _aniso_u;
    } else {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        assert(i != _alt_loc_map.end());
        a = (*i).second.create_aniso_u();
    }
    (*a)[0] = u11;
    (*a)[1] = u12;
    (*a)[2] = u13;
    (*a)[3] = u22;
    (*a)[4] = u23;
    (*a)[5] = u33;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_ANISO_U);
}

void
Atom::set_bfactor(float bfactor)
{
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        (*i).second.bfactor = bfactor;
    } else
        structure()->active_coord_set()->set_bfactor(this, bfactor);
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_BFACTOR);
}

void
Atom::set_color(const Rgba& rgba)
{
    if (rgba == _rgba)
        return;
    graphics_changes()->set_gc_color();
    // Don't include color change in modified list for speed with large structures, ticket #3000.
    change_tracker()->add_modified(structure(), static_cast<Atom*>(nullptr),
                                   ChangeTracker::REASON_COLOR);
    _rgba = rgba;
}

void
Atom::set_coord(const Coord& coord, CoordSet* cs, bool track_change)
{
    if (track_change) {
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_COORD);
        if (structure()->active_coord_set() == cs)
            structure()->change_tracker()->add_modified(structure(), structure(),
                ChangeTracker::REASON_SCENE_COORD);
    }
    if (cs == nullptr) {
        cs = structure()->active_coord_set();
        if (cs == nullptr) {
            if (structure()->coord_sets().size() > 0)
                cs = structure()->coord_sets()[0];
            else
                cs = structure()->new_coord_set();
            structure()->set_active_coord_set(cs);
        }
    }
    
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        (*i).second.coord = coord;
        if (_coord_index == COORD_UNASSIGNED)
            _coord_index = _new_coord(coord);
    } else
        _coordset_set_coord(coord, cs, track_change);
}

void
Atom::set_coord_index(unsigned int ci)
{
    if (_coord_index != COORD_UNASSIGNED)
        throw std::logic_error("Coordinate index already assigned");
    auto cs = structure()->active_coord_set();
    if (cs == nullptr)
        throw std::logic_error("Cannot assign coordinate index with no coordinate sets");
    if (cs->coords().size() <= ci)
        throw std::logic_error("Coordinate index larger than coordinate set");
    _coord_index = ci;
}

void
Atom::set_display(bool d)
{
    if (d == _display)
        return;
    graphics_changes()->set_gc_shape();
    graphics_changes()->set_gc_display();
    graphics_changes()->set_gc_ring();
    // Don't include display change in modified list for speed with large structures, ticket #3000.
    change_tracker()->add_modified(structure(), static_cast<Atom*>(nullptr),
                                   ChangeTracker::REASON_DISPLAY);
    _display = d;
}

void
Atom::set_draw_mode(DrawMode dm)
{
    if (dm == _draw_mode)
        return;
    graphics_changes()->set_gc_shape();
    graphics_changes()->set_gc_display();	// Sphere style can effect if bonds are shown.
    graphics_changes()->set_gc_ring();
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_DRAW_MODE);
    _draw_mode = dm;
}

void
Atom::set_hide(int h)
{
    if (h == _hide)
        return;
    graphics_changes()->set_gc_shape();
    graphics_changes()->set_gc_display();
    graphics_changes()->set_gc_ring();
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_HIDE);
    _hide = h;
}

void
Atom::set_occupancy(float occupancy)
{
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_OCCUPANCY);
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        (*i).second.occupancy = occupancy;
    } else
        structure()->active_coord_set()->set_occupancy(this, occupancy);
}

void
Atom::set_radius(float r)
{
    if (r <= 0.0)
        throw std::logic_error("radius must be positive");

    if (r == _radius)
        return;

    graphics_changes()->set_gc_shape();
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RADIUS);
    _radius = r;
}

void
Atom::clear_ribbon_coord() {
  if (_ribbon_coord) {
    delete _ribbon_coord;
    _ribbon_coord = nullptr;
  }
}

void
Atom::set_ribbon_coord(const Point& coord) {
  Real x = coord[0], y = coord[1], z = coord[2];
  if (_ribbon_coord)
    _ribbon_coord->set_xyz(x,y,z);
  else
    _ribbon_coord = new Coord(x,y,z);
}

void
Atom::set_selected(bool s)
{
    if (s == _selected)
        return;
    graphics_changes()->set_gc_select();
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_SELECTED);
    _selected = s;
}

void
Atom::set_serial_number(int sn)
{
    _serial_number = sn;
    if (_alt_loc != ' ') {
        _Alt_loc_map::iterator i = _alt_loc_map.find(_alt_loc);
        (*i).second.serial_number = sn;
    }
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_SERIAL_NUMBER);
}

std::vector<Atom*>
Atom::side_atoms(const Atom* skip_atom, const Atom* cycle_atom) const
// Returns all the atoms connected to this atom on "this side" of skip_atom (so connectivity will never
// trace thorugh skip_atom), considering missing structure as connecting; raises logic_error if we reach
// cycle_atom (so that Python throws ValueError); even if cycle_atom is skip_atom
{
    std::map<Atom*, std::vector<Atom*>> pb_connections;
    auto pbg = const_cast<Structure*>(structure())->pb_mgr().get_group(
        Structure::PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg != nullptr) {
        for (auto& pb: pbg->pseudobonds()) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            pb_connections[a1].push_back(a2);
            pb_connections[a2].push_back(a1);
        }
    }
    std::vector<Atom*> side_atoms;
    side_atoms.push_back(const_cast<Atom*>(this));
    std::set<const Atom*> seen;
    seen.insert(this);
    std::vector<Atom*> to_do;
    for (auto nb: neighbors())
        if (nb != skip_atom)
            to_do.push_back(nb);
    while (to_do.size() > 0) {
        auto a = to_do.back();
        to_do.pop_back();
        if (seen.find(a) != seen.end())
            continue;
        if (a == cycle_atom)
            throw std::logic_error("Atom::side_atoms() called on bond in ring or cycle");
        seen.insert(a);
        side_atoms.push_back(a);
        if (pb_connections.find(a) != pb_connections.end()) {
            for (auto conn: pb_connections[a]) {
                if (conn == cycle_atom)
                    throw std::logic_error("Atom::side_atoms() called on bond in ring or cycle");
                if (conn != skip_atom)
                    to_do.push_back(conn);
            }
        }
        for (auto nb: a->neighbors()) {
            if (nb == cycle_atom)
                throw std::logic_error("Atom::side_atoms() called on bond in ring or cycle");
            if (nb != skip_atom)
                to_do.push_back(nb);
        }
    }
    return side_atoms;
}

std::string
Atom::str() const
{
    std::string ret = residue()->str();
    ret += " ";
    ret += name();
    return ret;
}

}  // namespace atomstruct
