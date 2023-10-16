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

#ifndef templates_Residue
#define    templates_Residue

#include <map>
#include <pyinstance/PythonInstance.declare.h>
#include <vector>
#include "../imex.h"
#include "../string_types.h"
#include "../polymer.h"

namespace tmpl {

using atomstruct::AtomName;
using atomstruct::AtomType;
using atomstruct::ResName;
using atomstruct::PolymerType;

class Atom;
class Molecule;

class ATOMSTRUCT_IMEX Residue: public pyinstance::PythonInstance<Residue> {
    friend class Molecule;
    void    operator=(const Residue &);    // disable
    Residue(const Residue &);    // disable
    ~Residue();
public:
    std::vector<Atom*>  atoms() const;
    void    add_atom(Atom *element);
    Atom    *find_atom(const AtomName&) const;

    // return atoms that received assignments from the template
    std::vector<Atom *>    template_assign(
                  void (Atom::*assign_func)(const AtomType&),
                  const char *app,
                  const char *template_dir,
                  const char *extension
                ) const;

    const ResName&    name() const { return _name; }
    Atom    *chief() const { return _chief; }
    void    chief(Atom *a) { _chief = a; }
    Atom    *link() const { return _link; }
    void    link(Atom *a) { _link = a; }
    const std::string    &description() const { return _description; }
    void    description(const std::string &d) { _description = d; }
    PolymerType  polymer_type() const { return _polymer_type; }
    void    polymer_type(PolymerType pt) { _polymer_type = pt; }

    // alternative to chief/link for mmCIF templates
    typedef std::vector<Atom *> AtomList;
    void    add_link_atom(Atom *);
    const AtomList &link_atoms() const { return _link_atoms; }

    typedef std::map<AtomName, Atom *> AtomsMap;
    const AtomsMap &atoms_map() const { return _atoms; }
    bool        pdbx_ambiguous;      // for mmCIF ambiguous chemistry
    bool        has_metal() const { return _has_metal; }

    std::map<std::string, std::vector<std::string>> metadata;
private:
    Residue(Molecule *, const char *t);
    ResName     _name;
    AtomsMap    _atoms;
    Atom        *_chief, *_link;
    std::string _description;
    AtomList    _link_atoms;
    bool        _has_metal;
    PolymerType _polymer_type;
};

}  // namespace tmpl

#endif  // templates_Residue
