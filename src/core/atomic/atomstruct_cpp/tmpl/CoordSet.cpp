// vi: set expandtab ts=4 sw=4:
#include <algorithm>        // use std::find()
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
