#include <algorithm>		// use std::find()
#include "restmpl.h"

#ifdef UNPORTED
void
TmplCoordSet::fill(const TmplCoordSet *source) {
	size_t sourceSize = source->coords().size();
	while (Coords_.size() < sourceSize)
		Coords_.push_back(*source->findCoord(Coords_.size()));
}
#endif  // UNPORTED

void
TmplCoordSet::add_coord(TmplCoord element)
{
	_coords.push_back(element);
}
#ifdef UNPORTED
void
TmplCoordSet::removeCoord(TmplCoord *element)
{
	{
		std::vector<TmplCoord>::iterator i = std::find(Coords_.begin(), Coords_.end(), *element);
		if (i != Coords_.end())
			Coords_.erase(i);
	}
}
#endif  // UNPORTED
const TmplCoord *
TmplCoordSet::find_coord(size_t index) const
{
	if (index >= _coords.size())
		throw std::out_of_range("index out of range");
	return &_coords[index];
}
TmplCoord *
TmplCoordSet::find_coord(size_t index)
{
	if (index >= _coords.size())
		throw std::out_of_range("index out of range");
	return &_coords[index];
}
TmplCoordSet::TmplCoordSet(TmplMolecule *, int k): _csid(k)

{
}
#ifdef UNPORTED
TmplCoordSet::TmplCoordSet(TmplMolecule *, int k, int size): _csid(k)

{
	Coords_.reserve(size);
}
#endif  // UNPORTED

TmplCoordSet::~TmplCoordSet()
{
}

