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

#ifndef templates_Atom
#define    templates_Atom

#include "../imex.h"

#include <vector>
#include "CoordSet.h"
#include <element/Element.h>
#include <pyinstance/PythonInstance.declare.h>
#include "../string_types.h"

namespace tmpl {

using atomstruct::AtomName;
using atomstruct::AtomType;
using element::Element;

class Molecule;
class Residue;
class Bond;

class ATOMSTRUCT_IMEX Atom: public pyinstance::PythonInstance<Atom> {
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
    char          _chirality;
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
    void          add_bond(Bond *b);
    const Bonds&  bonds() const { return _bonds; }
    char          chirality() const { return _chirality; }
    const Coord&  coord() const;
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
    Atom(Molecule *, const AtomName& n, const Element &e, char chirality);
};

}  // namespace tmpl

#include "Bond.h"

namespace tmpl {
    
inline void
Atom::add_bond(Bond *b) {
    _bonds.push_back(b);
    _neighbors.push_back(b->other_atom(this));
}

}

#endif  // templates_Atom
