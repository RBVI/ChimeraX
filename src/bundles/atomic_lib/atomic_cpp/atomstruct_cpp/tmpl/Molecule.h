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

#ifndef templates_Molecule
#define    templates_Molecule

#include <set>
#include <map>
#include <vector>
#include <string>
#include <pyinstance/PythonInstance.declare.h>

#include "../imex.h"
#include "TAexcept.h"
#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "Residue.h"
#include "../string_types.h"

namespace tmpl {

using atomstruct::ResName;

class ATOMSTRUCT_IMEX Molecule: public pyinstance::PythonInstance<Molecule> {
public:
        ~Molecule();
    Atom    *new_atom(const AtomName& n, const element::Element& e, char chirality = '?');
    typedef std::set<Atom *> Atoms;
    typedef std::set<Bond *> Bonds;
    typedef std::vector<CoordSet *> CoordSets;
    typedef std::map<ResName, Residue *> Residues;
private:
    Atoms    _atoms;
    Bonds    _bonds;
    CoordSets    _coord_sets;
    Residues    _residues;
public:
    Bond    *new_bond(Atom *a0, Atom *a1);
    CoordSet    *new_coord_set(int key);
    const CoordSets    &coord_sets() const { return _coord_sets; }
    CoordSet    *find_coord_set(int) const;
    Residue    *new_residue(const char *t);
    void        delete_residue(Residue* r);
    Residue    *find_residue(const ResName&) const;
    void        set_active_coord_set(CoordSet *cs);
    CoordSet    *active_coord_set() const { return _active_cs; }
    const Residues &residues_map() { return _residues; }
private:
    CoordSet    *_active_cs;
public:
    Molecule();
};

}  // namespace tmpl

#endif  // templates_Molecule
