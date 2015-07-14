// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Atom
#define atomstruct_Atom

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <basegeom/Coord.h>
#include <basegeom/Point.h>
#include <basegeom/Sphere.h>
#include "Element.h"
#include "imex.h"
#include "string_types.h"

namespace atomstruct {

using basegeom::GraphicsContainer;
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
        unsigned int  substituents;
        std::string  description;
    };
    typedef std::map<AtomType, IdatmInfo> IdatmInfoMap;
    typedef std::vector<const Ring*>  Rings;

private:
    static const unsigned int  COORD_UNASSIGNED = ~0u;
    Atom(AtomicStructure *as, const char* name, Element e);
    char  _alt_loc;
    typedef struct {
        std::vector<float> *  aniso_u;
        float  bfactor;
        Point  coord;
        float  occupancy;
        int  serial_number;
    } _Alt_loc_info;
    typedef std::map<unsigned char, _Alt_loc_info>  _Alt_loc_map;
    _Alt_loc_map  _alt_loc_map;
    std::vector<float> *  _aniso_u;
    mutable AtomType  _computed_idatm_type;
    unsigned int  _coord_index;
    void  _coordset_set_coord(const Point &);
    void  _coordset_set_coord(const Point &, CoordSet *cs);
    Element  _element;
    AtomType  _explicit_idatm_type;
    AtomName  _name;
    unsigned int  _new_coord(const Point &);
    float  _radius = -1.0; // indicates not explicitly set
    Residue *  _residue;
    mutable Rings  _rings;
    int  _serial_number;
    AtomicStructure *  _structure;
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
    int  coordination(int value_if_unknown) const;
    virtual const basegeom::Coord &coord() const;
    float  default_radius() const;
    Element  element() const { return _element; }
    static const IdatmInfoMap&  get_idatm_info_map();
    bool  has_alt_loc(char al) const
      { return _alt_loc_map.find(al) != _alt_loc_map.end(); }
    bool  idatm_is_explicit() const { return _explicit_idatm_type[0] != '\0'; }
    const AtomType&  idatm_type() const;
    const AtomName&  name() const { return _name; }
    // neighbors() just simply inherited from Connectible (via BaseSphere)
    float  occupancy() const;
    int  serial_number() const { return _serial_number; }
    float  radius() const;
    void  register_field(std::string /*name*/, int /*value*/) {}
    void  register_field(std::string /*name*/, double /*value*/) {}
    void  register_field(std::string /*name*/, const std::string &/*value*/) {}
    void  remove_bond(Bond *b) { remove_connection(b); }
    Residue *  residue() const { return _residue; }
    const Rings&  rings(bool cross_residues = false, int all_size_threshold = 0,
            std::set<const Residue*>* ignore = nullptr) const;
    void  set_alt_loc(char alt_loc, bool create=false, bool from_residue=false);
    void  set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33);
    void  set_bfactor(float);
    virtual void  set_coord(const Point & coord) { set_coord(coord, NULL); }
    void  set_coord(const Point & coord, CoordSet * cs);
    void  set_computed_idatm_type(const char* it) { _computed_idatm_type =  it; }
    void  set_idatm_type(const char* it) { _explicit_idatm_type = it; }
    void  set_idatm_type(const std::string& it) { set_idatm_type(it.c_str()); }
    void  set_occupancy(float);
    void  set_radius(float);
    void  set_serial_number(int);
    std::string  str() const;
    AtomicStructure*  structure() const { return _structure; }

    // graphics related
    GraphicsContainer*  graphics_container() const {
        return reinterpret_cast<GraphicsContainer*>(_structure); }
};

}  // namespace atomstruct

#include "AtomicStructure.h"
inline const atomstruct::AtomType&
atomstruct::Atom::idatm_type() const {
    if (idatm_is_explicit()) return _explicit_idatm_type;
    if (!_structure->_idatm_valid) _structure->_compute_idatm_types();
    return _computed_idatm_type;
}

#endif  // atomstruct_Atom
