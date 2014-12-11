// vim: set expandtab ts=4 sw=4:
#ifndef atomstruct_Atom
#define atomstruct_Atom

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>

#include "Element.h"
#include <basegeom/Point.h>
#include <basegeom/Coord.h>
#include <basegeom/Sphere.h>
#include "imex.h"

namespace atomstruct {

using basegeom::Point;

class AtomicStructure;
class Bond;
class CoordSet;
class Residue;
class Ring;

class ATOMSTRUCT_IMEX Atom: public basegeom::BaseSphere<Bond, Atom> {
    friend class AtomicStructure;
    friend class Residue;
public:
    typedef Connections Bonds;
    enum IdatmGeometry {
        Ion=0, Single=1, Linear=2, Planar=3, Tetrahedral=4
    };
    struct IdatmInfo {
        IdatmGeometry  geometry;
        int  substituents;
        std::string  description;
    };
    typedef std::unordered_map<std::string, IdatmInfo> IdatmInfoMap;
    typedef std::vector<const Ring*>  Rings;

private:
    static const unsigned int  COORD_UNASSIGNED = ~0u;
    Atom(AtomicStructure *as, const std::string &name, Element e);
    char  _alt_loc;
    typedef struct {
        std::vector<float> *  aniso_u;
        float  bfactor;
        Point  coord;
        float  occupancy;
        int  serial_number;
    } _Alt_loc_info;
    typedef std::unordered_map<unsigned char, _Alt_loc_info>  _Alt_loc_map;
    _Alt_loc_map  _alt_loc_map;
    mutable std::string  _computed_idatm_type;
    unsigned int  _coord_index;
    Element  _element;
    std::string  _explicit_idatm_type;
    AtomicStructure *  _structure;
    std::string  _name;
    Residue *  _residue;
    std::vector<float> *  _aniso_u;
    mutable Rings  _rings;
    int  _serial_number;
    void  _coordset_set_coord(const Point &);
    void  _coordset_set_coord(const Point &, CoordSet *cs);
    unsigned int  _new_coord(const Point &);
public:
    // so that I/O routines can cheaply "change their minds" about element
    // types during early structure creation
    void  _switch_initial_element(Element e) { _element = e; }

public:
    void  add_bond(Bond *b) { add_connection(b); }
    char  alt_loc() const { return _alt_loc; }
    std::set<char>  alt_locs() const;
    float  bfactor() const;
    const Bonds&  bonds() const { return connections(); }
    // connects_to() just simply inherited from Connectible (via BaseSphere)
    unsigned int  coord_index() const { return _coord_index; }
    virtual const basegeom::Coord &coord() const;
    Element  element() const { return _element; }
    static const IdatmInfoMap&  get_idatm_info_map();
    bool  has_alt_loc(char al) const
      { return _alt_loc_map.find(al) != _alt_loc_map.end(); }
    bool  idatm_is_explicit() const { return !_explicit_idatm_type.empty(); }
    const std::string&  idatm_type() const;
    const std::string  name() const { return _name; }
    // neighbors() just simply inherited from Connectible (via BaseSphere)
    float  occupancy() const;
    int  serial_number() const { return _serial_number; }
    void  register_field(std::string name, int value) {}
    void  register_field(std::string name, double value) {}
    void  register_field(std::string name, const std::string value) {}
    void  remove_bond(Bond *b) { remove_connection(b); }
    Residue *  residue() const { return _residue; }
    const Rings&  rings(bool cross_residues = false, int all_size_threshold = 0,
            std::set<const Residue*>* ignore = nullptr) const;
    void  set_alt_loc(char alt_loc, bool create=false, bool from_residue=false);
    void  set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33);
    void  set_bfactor(float);
    virtual void  set_coord(const Point & coord) { set_coord(coord, NULL); }
    void  set_coord(const Point & coord, CoordSet * cs);
    void  set_computed_idatm_type(const std::string& it) {
        _computed_idatm_type = it;
    }
    void  set_idatm_type(const std::string& it) { _explicit_idatm_type = it; }
    void  set_occupancy(float);
    void  set_serial_number(int);
    AtomicStructure *  structure() const { return _structure; }
};

}  // namespace atomstruct

#include "AtomicStructure.h"
inline const std::string&
atomstruct::Atom::idatm_type() const {
    if (idatm_is_explicit()) return _explicit_idatm_type;
    if (!_structure->_idatm_valid) _structure->_compute_idatm_types();
    return _computed_idatm_type;
}

#endif  // atomstruct_Atom
