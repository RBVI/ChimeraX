// vim: set expandtab ts=4 sw=4:
#include <algorithm>        // use std::find()
#include "restmpl.h"

void
TmplCoordSet::add_coord(TmplCoord element)
{
    _coords.push_back(element);
}
const TmplCoord *
TmplCoordSet::find_coord(std::size_t index) const
{
    if (index >= _coords.size())
        throw std::out_of_range("index out of range");
    return &_coords[index];
}
TmplCoord *
TmplCoordSet::find_coord(std::size_t index)
{
    if (index >= _coords.size())
        throw std::out_of_range("index out of range");
    return &_coords[index];
}
TmplCoordSet::TmplCoordSet(TmplMolecule *, int k): _csid(k)

{
}

TmplCoordSet::~TmplCoordSet()
{
}

