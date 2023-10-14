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

#include <algorithm>        // use std::find()
#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "restmpl.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<tmpl::CoordSet>;

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
