// vim: set expandtab ts=4 sw=4:
#ifndef atomic_Atom
#define atomic_Atom

#include <vector>
#include <string>
#include <map>
#include <set>

#include "Element.h"
#include "base-geom/Point.h"
#include "base-geom/Coord.h"
#include "Bond.h"
#include "base-geom/Sphere.h"
#include "imex.h"

class CoordSet;
class AtomicStructure;
class Residue;

class ATOMSTRUCT_IMEX Atom: public BaseSphere<Bond, Atom> {
    friend class AtomicStructure;
    friend class Residue;
public:
    typedef ConnectionsMap BondsMap;
    typedef Connections Bonds;

private:
    static const unsigned int  COORD_UNASSIGNED = ~0u;
    Atom(AtomicStructure *as, std::string &name, Element e);
    unsigned int  _coord_index;
    Element  _element;
    AtomicStructure *  _structure;
    std::string  _name;
    Residue *  _residue;
    typedef struct {
        std::vector<float> *  aniso_u;
        float  bfactor;
        Point  coord;
        float  occupancy;
        int  serial_number;
    } _Alt_loc_info;
    typedef std::map<char, _Alt_loc_info>  _Alt_loc_map;
    _Alt_loc_map  _alt_loc_map;
    char  _alt_loc;
    std::vector<float> *  _aniso_u;
    int  _serial_number;
    void  _coordset_set_coord(const Point &);
    void  _coordset_set_coord(const Point &, CoordSet *cs);
    unsigned int  _new_coord(const Point &);

public:
    void  add_bond(Bond *b) { add_connection(b); }
    char  alt_loc() const { return _alt_loc; }
    std::set<char>  alt_locs() const;
    float  bfactor() const;
    Bonds  bonds() const { return connections(); }
    const BondsMap &    bonds_map() const { return connections_map(); }
    // connects_to() just simply inherited from Connectible (via BaseSphere)
    unsigned int  coord_index() const { return _coord_index; }
    virtual const Coord &coord() const;
    Element  element() const { return _element; }
    bool  has_alt_loc(char al) const
      { return _alt_loc_map.find(al) != _alt_loc_map.end(); }
    const std::string  name() const { return _name; }
    float  occupancy() const;
    int  serial_number() const { return _serial_number; }
    void  register_field(std::string name, int value) {}
    void  register_field(std::string name, double value) {}
    void  register_field(std::string name, const std::string value) {}
    void  remove_bond(Bond *b) { remove_connection(b); }
    Residue *  residue() const { return _residue; }
    void  set_alt_loc(char alt_loc, bool create=false, bool from_residue=false);
    void  set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33);
    void  set_bfactor(float);
    virtual void  set_coord(const Point & coord) { set_coord(coord, NULL); }
    void  set_coord(const Point & coord, CoordSet * cs);
    void  set_occupancy(float);
    void  set_serial_number(int);
    AtomicStructure *  structure() const { return _structure; }
};

#endif  // atomic_Atom
