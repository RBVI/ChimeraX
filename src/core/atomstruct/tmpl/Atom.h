// vim: set expandtab ts=4 sw=4:
#ifndef templates_Atom
#define    templates_Atom

#include <vector>
#include "CoordSet.h"
#include "../Element.h"
#include "../imex.h"
#include "Bond.h"

namespace tmpl {

class Molecule;
class Residue;
using atomstruct::Element;

class ATOMSTRUCT_IMEX Atom {
public:
    typedef std::vector<Bond*> Bonds;
    typedef std::vector<Atom*> Neighbors;
private:
    friend class Molecule;
    friend class Residue;
    void    operator=(const Atom &);    // disable
        Atom(const Atom &);    // disable
        ~Atom();
    Bonds         _bonds;
    Element       _element;
    Molecule*     _molecule;
    std::string   _name;
    Neighbors     _neighbors;
    Residue*      _residue;
public:
    void          add_bond(Bond *b) {
        _bonds.push_back(b);
        _neighbors.push_back(b->other_atom(this));
    }
    const Bonds&  bonds() const { return _bonds; }
    Element       element() const { return _element; }
    Molecule*     molecule() const { return _molecule; }
    Residue*      residue() const { return _residue; }
    std::string   name() const { return _name; }
    const Neighbors&    neighbors() const { return _neighbors; }
public:
    static const unsigned int COORD_UNASSIGNED = ~0u;
    void        set_coord(const Coord &c);
    void        set_coord(const Coord &c, CoordSet *cs);
private:
    mutable unsigned int _index;
    int    new_coord(const Coord &c) const;
private:
    std::string     _idatm_type;
public:
    std::string  idatm_type() const { return _idatm_type; }
    void    set_idatm_type(const char *i) { _idatm_type = i; }
private:
    Atom(Molecule *, std::string &n, atomstruct::Element e);
};

}  // namespace tmpl

#endif  // templates_Atom
