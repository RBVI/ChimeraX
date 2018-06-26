// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef templates_CoordSet
#define    templates_CoordSet

#include <vector>
#include "../imex.h"
#include "Coord.h"

namespace tmpl {

class Molecule;

class ATOMSTRUCT_IMEX CoordSet {
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
