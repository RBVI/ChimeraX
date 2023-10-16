// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_Atom
#define atomstruct_Atom

#include <algorithm>  // std::find
#include <cstring>
#include <element/Element.h>
#include <map>
#include <math.h> // isnan
#include <memory>
#include <pyinstance/PythonInstance.declare.h>
#include <set>
#include <string>
#include <vector>

#include "backbone.h"
#include "ChangeTracker.h"
#include "Coord.h"
#include "imex.h"
#include "Point.h"
#include "Rgba.h"
#include "session.h"
#include "string_types.h"
#include "Structure.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

using element::Element;
    
namespace atomstruct {

class Bond;
class CoordSet;
class Residue;
class Ring;

class ATOMSTRUCT_IMEX Atom: public pyinstance::PythonInstance<Atom>  {
    friend class AtomicStructure;
    friend class UniqueConnection;
    friend class Structure;
    friend class Residue;
    friend class StructurePBGroup;
    friend class CS_PBGroup;
public:
    // HIDE_ constants are masks for hide bits
    static const unsigned int  HIDE_RIBBON = 0x1;
    static const unsigned int  HIDE_ISOLDE = 0x2;
    static const unsigned int  HIDE_NUCLEOTIDE = 0x4;

    typedef std::vector<Bond*> Bonds;
    enum DrawMode: unsigned char { Sphere, Ball, EndCap };
    enum IdatmGeometry { Ion=0, Single=1, Linear=2, Planar=3, Tetrahedral=4 };
    struct IdatmInfo {
        IdatmGeometry  geometry;
        unsigned int  substituents;
        std::string  description;
    };
    typedef std::map<AtomType, IdatmInfo> IdatmInfoMap;
    typedef std::vector<Atom*>  Neighbors;
    typedef std::vector<const Ring*>  Rings;
    enum class StructCat { Unassigned, Main, Ligand, Ions, Solvent };

    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 10; };
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 1; };
    static int  SESSION_ALTLOC_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 3; };
    static int  SESSION_ALTLOC_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 5; };
private:
    static const unsigned int  COORD_UNASSIGNED = ~0u;
    Atom(Structure *as, const char* name, const Element& e);
    virtual ~Atom();

    char  _alt_loc;
    class _Alt_loc_info {
      public:
        _Alt_loc_info() : bfactor(0.0), serial_number(0) {}
        ~_Alt_loc_info() { }

        std::vector<float> *create_aniso_u() {
            if (aniso_u.get() == nullptr) {
                aniso_u = std::make_shared<std::vector<float>>();
                aniso_u.get()->resize(6);
            }
            return aniso_u.get();
        }

        std::shared_ptr<std::vector<float>>  aniso_u;
        float  bfactor;
        Point  coord;
        float  occupancy;
        int  serial_number;
    };
    typedef std::map<unsigned char, _Alt_loc_info>  _Alt_loc_map;
    _Alt_loc_map  _alt_loc_map;
    std::vector<float>*  _aniso_u;
    Bonds  _bonds; // _bonds/_neighbors in same order
    mutable AtomType  _computed_idatm_type;
    unsigned int  _coord_index;
    void  _coordset_set_coord(const Point &, CoordSet *cs, bool track_change);
    bool  _display = true;
    bool  _in_ribbon = false;
    DrawMode  _draw_mode = DrawMode::Sphere;
    const Element*  _element;
    AtomType  _explicit_idatm_type;
    int  _hide = 0;
    AtomName  _name;
    Neighbors  _neighbors; // _bonds/_neighbors in same order
    unsigned int  _new_coord(const Point &);
    mutable float  _radius;
    Residue *  _residue;
    Rgba  _rgba;
    Coord *_ribbon_coord;
    mutable Rings  _rings;
    bool  _selected = false;
    int  _serial_number;
    void  _set_structure_category(Atom::StructCat sc) const;
    Structure*  _structure;
    mutable StructCat  _structure_category;
    void  _uncache_radius() const { if (_radius < 0.0) _radius = 0.0; }
public:
    // so that I/O routines can cheaply "change their minds" about element
    // types during early structure creation
    void  _switch_initial_element(const Element& e) { _element = &e; }

public:
    void  add_bond(Bond *b);
    char  alt_loc() const { return _alt_loc; }
    std::set<char>  alt_locs() const;
    const std::vector<float> *aniso_u() const;
    const std::vector<float> *aniso_u(char alt_loc) const;
    float  bfactor() const;
    float  bfactor(char alt_loc) const { return _alt_loc_map.find(alt_loc)->second.bfactor; }
    const Bonds&  bonds() const { return _bonds; }
    void  clean_alt_locs();
    void  clear_aniso_u();
    void  clear_ribbon_coord();
    bool  connects_to(const Atom* other) const {
        return std::find(_neighbors.begin(), _neighbors.end(), other) != _neighbors.end();
    }
    const Coord&  coord() const;
    const Coord&  coord(const CoordSet* cs) const;
    const Coord&  coord(char alt_loc) const { return _alt_loc_map.find(alt_loc)->second.coord; }
    unsigned int  coord_index() const { return _coord_index; }
    int  coordination(int value_if_unknown) const;
    float  default_radius() const;
    void  delete_alt_loc(char al);
    DrawMode  draw_mode() const { return _draw_mode; }
    const Element&  element() const { return *_element; }
    static const IdatmInfoMap&  get_idatm_info_map();
    bool  has_alt_loc(char al) const
      { return _alt_loc_map.find(al) != _alt_loc_map.end(); }
    bool  has_aniso_u() const { return aniso_u() != nullptr; }
    bool  has_aniso_u(char alt_loc) const { return aniso_u(alt_loc) != nullptr; }
    bool  has_missing_structure_pseudobond() const;
    bool  idatm_is_explicit() const { return _explicit_idatm_type[0] != '\0'; }
    const AtomType&  idatm_type() const;
    bool  is_backbone(BackboneExtent bbe) const;
    bool  is_missing_heavy_template_neighbors(bool chain_start = false, bool chain_end = false,
        bool no_template_okay = false) const;
    bool  is_ribose() const;
    bool  is_side_connector() const;
    bool  is_side_chain(bool only) const;
    const AtomName&  name() const { return _name; }
    const Neighbors&  neighbors() const { return _neighbors; }
    Bonds::size_type  num_explicit_bonds() const; // includes missing-structure bonds
    float  occupancy() const;
    float  occupancy(char alt_loc) const { return _alt_loc_map.find(alt_loc)->second.occupancy; }
    float radius() const {
        if (_radius == 0.0) {
            auto r = default_radius();
            _radius = 0.0 - r;
            return r;
        }
        if (_radius < 0.0) // cached from default computation
            return 0.0 - _radius;
        // explicitly set
        return _radius;
    }
    float maximum_bond_radius(float default_radius) const;
    void  remove_bond(Bond *b);
    Residue *  residue() const { return _residue; }
    const Coord *ribbon_coord() const { return _ribbon_coord; }
    Coord  effective_coord() const;
    Coord  effective_scene_coord() const;
    const Rings&  rings(bool cross_residues = false, int all_size_threshold = 0,
            std::set<const Residue*>* ignore = nullptr) const;
    Coord  scene_coord() const;
    Coord  scene_coord(const CoordSet* cs) const;
    Coord  scene_coord(char alt_loc) const;
    int  serial_number() const { return _serial_number; }
    int  session_num_ints(int version=CURRENT_SESSION_VERSION) const {
        return SESSION_NUM_INTS(version) + Rgba::session_num_ints()
            + _alt_loc_map.size() * SESSION_ALTLOC_INTS(version);
    }
    int  session_num_floats(int version=CURRENT_SESSION_VERSION) const;
    void  session_restore(int version, int** ints, float** floats, PyObject* misc);
    void  session_save(int** ints, float** floats, PyObject* misc) const;
    void  set_alt_loc(char alt_loc, bool create=false, bool _from_residue=false);
    void  set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33);
    void  set_bfactor(float);
    void  set_coord(const Point& coord) { set_coord(coord, nullptr, true); }
    void  set_coord(const Point& coord, CoordSet* cs) { set_coord(coord, cs, true); }
    void  set_coord(const Point& coord, bool track_change) {
        set_coord(coord, nullptr, track_change);
    }
    void  set_coord(const Point& coord, CoordSet* cs, bool track_change);
    void  set_coord_index(unsigned int);
    void  set_computed_idatm_type(const char* it);
    void  set_draw_mode(DrawMode dm);
    void  set_element(const Element& e);
    void  set_idatm_type(const char* it);
    void  set_idatm_type(const std::string& it) { set_idatm_type(it.c_str()); }
    void  set_implicit_idatm_type(const char* it);
    void  set_implicit_idatm_type(const std::string& it) { set_implicit_idatm_type(it.c_str()); }
    void  set_name(const AtomName& name);
    void  set_occupancy(float);
    void  set_radius(float);
    void  set_ribbon_coord(const Point& coord);
    void  set_serial_number(int);
    std::vector<Atom*>  side_atoms(const Atom* skip_neighbor, const Atom* cycle_atom) const;
    std::string  str() const;
    Structure*  structure() const { return _structure; }
    StructCat  structure_category() const;
    void  use_default_radius() { _radius = 0.0; }

    // change tracking
    ChangeTracker*  change_tracker() const;

    // graphics related
    void  clear_hide_bits(int bit_mask) { set_hide(hide() & ~bit_mask); }
    const Rgba&  color() const { return _rgba; }
    bool  display() const { return _display; }
    int   hide() const { return _hide; }
    bool  in_ribbon() const { return _in_ribbon; }
    GraphicsChanges*  graphics_changes() const {
        return reinterpret_cast<GraphicsChanges*>(structure()); }
    bool  selected() const { return _selected; }
    void  set_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b, Rgba::Channel a) {
        set_color(Rgba({r, g, b, a}));
    }
    void  set_color(const Rgba& rgba);
    void  set_display(bool d);
    void  set_hide(int h);
    void  set_hide_bits(int bit_mask) { set_hide(hide() | bit_mask); }
    void  set_in_ribbon(bool ir) { _in_ribbon = ir; }
    void  set_selected(bool s);
    bool  visible() const { return _display && !_hide; }
};

}  // namespace atomstruct

#include "Structure.h"

namespace atomstruct {

inline ChangeTracker*
Atom::change_tracker() const { return structure()->change_tracker(); }

inline const atomstruct::AtomType&
Atom::idatm_type() const {
    if (idatm_is_explicit()) return _explicit_idatm_type;
    structure()->ready_idatm_types();
    return _computed_idatm_type;
}

inline void
Atom::_set_structure_category(Atom::StructCat sc) const
{
    if (sc == _structure_category)
        return;
    change_tracker()->add_modified(structure(), const_cast<Atom*>(this),
        ChangeTracker::REASON_STRUCTURE_CATEGORY);
    _structure_category = sc;
}

inline void
Atom::set_computed_idatm_type(const char* it) {
    if (structure()->_atom_types_notify && !idatm_is_explicit() && _computed_idatm_type != it) {
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_IDATM_TYPE);
    }
    _computed_idatm_type =  it;
}

inline void
Atom::set_element(const Element& e) {
    change_tracker()->add_modified(structure(), const_cast<Atom*>(this), ChangeTracker::REASON_ELEMENT);
    _element = &e;
    _uncache_radius();
    _structure->_idatm_valid = false;
}

inline void
Atom::set_idatm_type(const char* it) {
    // make sure it actually is effectively different before tracking
    // change
    if (!(_explicit_idatm_type.empty() && _computed_idatm_type == it)
    && !(*it == '\0' && _explicit_idatm_type == _computed_idatm_type)
    && !(!_explicit_idatm_type.empty() && it == _explicit_idatm_type)) {
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_IDATM_TYPE);
        structure()->set_idatm_valid(false);
    }
    _explicit_idatm_type = it;
}

inline void
Atom::set_implicit_idatm_type(const char* it) {
    // used for setting the IDATM type but allowing it to be overridden by later recomputations
    // (usually after a modification)

    // run the computation first if it hasn't been run
    structure()->ready_idatm_types();

    // make sure it actually is effectively different before doing anything
    if (_explicit_idatm_type == it) {
        _explicit_idatm_type.clear();
        _computed_idatm_type = it;
        return;
    }
    if (_explicit_idatm_type.empty()) {
        if (_computed_idatm_type == it)
            return;
    } else {
        _explicit_idatm_type.clear();
    }
    _computed_idatm_type = it;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_IDATM_TYPE);
}

inline void
Atom::set_name(const AtomName& name) {
    if (name == _name)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_NAME);
    _name = name;
}

inline Atom::StructCat
Atom::structure_category() const {
    if (structure()->_structure_cats_dirty) structure()->_compute_structure_cats();
    return _structure_category;
}

}  // namespace atomstruct

#endif  // atomstruct_Atom
