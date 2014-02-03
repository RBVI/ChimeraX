// vim: set expandtab ts=4 sw=4:
#ifndef molecule_Atom
#define molecule_Atom

#include <vector>
#include <string>
#include <map>

#include "Element.h"
#include "base-geom/Point.h"
#include "base-geom/Coord.h"
#include "Bond.h"
#include "base-geom/Sphere.h"
#include "imex.h"

class CoordSet;
class Molecule;
class Residue;

class MOLECULE_IMEX Atom: public BaseSphere<Bond, Atom> {
    friend class Molecule;
    friend class Residue;
public:
    typedef ConnectionsMap BondsMap;
    typedef Connections Bonds;

private:
    static const unsigned int  COORD_UNASSIGNED = ~0u;
    Atom(Molecule *m, std::string &name, Element e);
    unsigned int  _coord_index;
    Element  _element;
    Molecule *  _molecule;
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
    float  bfactor() const;
    Bonds  bonds() const { return connections(); }
    const BondsMap &    bonds_map() const { return connections_map(); }
    // connects_to() just simply inherited from Connectible (via BaseSphere)
    unsigned int  coord_index() const { return _coord_index; }
    virtual const Coord &coord() const;
    Element  element() const { return _element; }
    Molecule *  molecule() const { return _molecule; }
    const std::string  name() const { return _name; }
    float  occupancy() const;
    void  register_field(std::string name, int value) {}
    void  register_field(std::string name, double value) {}
    void  register_field(std::string name, const std::string value) {}
    void  remove_bond(Bond *b) { remove_connection(b); }
    Residue *  residue() const { return _residue; }
    void  set_alt_loc(char alt_loc, bool create=false);
    void  set_aniso_u(float u11, float u12, float u13, float u22, float u23, float u33);
    void  set_bfactor(float);
    virtual void  set_coord(const Point & coord) { set_coord(coord, NULL); }
    void  set_coord(const Point & coord, CoordSet * cs);
    void  set_occupancy(float);
    void  set_serial_number(int);
};

#endif  // molecule_Atom
