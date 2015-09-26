// vi: set expandtab ts=4 sw=4:
#ifndef templates_Atom
#define    templates_Atom

#include <vector>
#include "Bond.h"
#include "CoordSet.h"
#include <element/Element.h>
#include "../imex.h"
#include "../string_types.h"

namespace tmpl {

using atomstruct::AtomName;
using atomstruct::AtomType;
using element::Element;

class Molecule;
class Residue;

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
    const Element*       _element;
    AtomType      _idatm_type;
    mutable unsigned int _index;
    Molecule*     _molecule;
    AtomName      _name;
    Neighbors     _neighbors;
    Residue*      _residue;
private:
    int    new_coord(const Coord &c) const;
public:
    void          add_bond(Bond *b) {
        _bonds.push_back(b);
        _neighbors.push_back(b->other_atom(this));
    }
    const Bonds&  bonds() const { return _bonds; }
    const Element&       element() const { return *_element; }
    Molecule*     molecule() const { return _molecule; }
    Residue*      residue() const { return _residue; }
    const AtomName&     name() const { return _name; }
    const Neighbors&    neighbors() const { return _neighbors; }
public:
    static const unsigned int COORD_UNASSIGNED = ~0u;
    void        set_coord(const Coord &c);
    void        set_coord(const Coord &c, CoordSet *cs);
public:
    const AtomType&  idatm_type() const { return _idatm_type; }
    void    set_idatm_type(const AtomType& i) { _idatm_type = i; }
private:
    Atom(Molecule *, const AtomName& n, const Element &e);
};

}  // namespace tmpl

#endif  // templates_Atom
