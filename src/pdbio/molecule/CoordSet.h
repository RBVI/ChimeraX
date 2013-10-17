#ifndef molecule_CoordSet
#define molecule_CoordSet

#include "base-geom/Coord.h"

#include <map>
#include <string>
#include <vector>

class CoordSet {
	friend class Atom;
	friend class Molecule;

public:
	typedef std::vector<Coord>  Coords;

private:
	Coords  _coords;
	int  _cs_id;
	std::map<const Atom *, float>  _bfactor_map;
	std::map<const Atom *, float>  _occupancy_map;
	CoordSet(int cs_id);
	CoordSet(int cs_id, int size);

public:
	void  add_coord(const Point &coord) { _coords.push_back(coord); }
	const Coords &  coords() const { return _coords; }
	float  get_bfactor(const Atom *) const;
	float  get_occupancy(const Atom *) const;
	void  fill(const CoordSet *source) { _coords = source->_coords; }
	int  id() const { return _cs_id; }
	void  set_bfactor(const Atom *, float);
	void  set_occupancy(const Atom *, float);
};

#endif  // molecule_CoordSet
