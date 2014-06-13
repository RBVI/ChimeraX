// vim: set expandtab ts=4 sw=4:
#ifndef templates_Atom
#define    templates_Atom

#include <map>
#include <vector>
#include "CoordSet.h"
#include "../Element.h"
#include "../imex.h"

namespace tmpl {

class Molecule;
class Residue;
class Bond;

class ATOMSTRUCT_IMEX Atom {
    friend class Molecule;
    friend class Residue;
    void    operator=(const Atom &);    // disable
        Atom(const Atom &);    // disable
        ~Atom();
    Molecule    *_molecule;
    Residue    *_residue;
public:
    void    add_bond(Bond *b);
    typedef std::map<Atom*, Bond *> BondsMap;
    const BondsMap    &bonds_map() const { return _bonds; }
    Molecule    *molecule() const { return _molecule; }
    Residue    *residue() const { return _residue; }
    std::string        name() const { return _name; }
private:
    std::string    _name;
    atomstruct::Element    _element;
    BondsMap    _bonds;
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
