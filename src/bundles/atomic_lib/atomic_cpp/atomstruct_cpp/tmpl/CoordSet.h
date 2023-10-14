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

#ifndef templates_CoordSet
#define    templates_CoordSet

#include <pyinstance/PythonInstance.declare.h>
#include <vector>
#include "../imex.h"
#include "Coord.h"

namespace tmpl {

class Molecule;

class ATOMSTRUCT_IMEX CoordSet: public pyinstance::PythonInstance<CoordSet> {
    friend class Molecule;
    void    operator=(const CoordSet &);    // disable
        CoordSet(const CoordSet &);    // disable
        ~CoordSet();
    std::vector<Coord>    _coords;
public:
    void    add_coord(Coord element);
    typedef std::vector<Coord> Coords;
    const Coords    &coords() const { return _coords; }
    const Coord    *find_coord(std::size_t) const;
    Coord    *find_coord(std::size_t);
public:
    int        id() const { return _csid; }
private:
    int    _csid;
private:
    CoordSet(Molecule *, int key);
};

}  // namespace tmpl

#endif  // templates_CoordSet
