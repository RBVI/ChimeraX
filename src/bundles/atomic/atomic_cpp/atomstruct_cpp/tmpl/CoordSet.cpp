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

#include <algorithm>        // use std::find()
#define ATOMSTRUCT_EXPORT
#include "restmpl.h"

namespace tmpl {

void
CoordSet::add_coord(Coord element)
{
    _coords.push_back(element);
}
const Coord *
CoordSet::find_coord(std::size_t index) const
{
    if (index >= _coords.size())
        throw std::out_of_range("index out of range");
    return &_coords[index];
}
Coord *
CoordSet::find_coord(std::size_t index)
{
    if (index >= _coords.size())
        throw std::out_of_range("index out of range");
    return &_coords[index];
}
CoordSet::CoordSet(Molecule *, int k): _csid(k)

{
}

CoordSet::~CoordSet()
{
}

}  // namespace tmpl
