// vim: set expandtab ts=4 sw=4:
#ifndef atomic_CoordSet
#define atomic_CoordSet

#include "basegeom/Coord.h"
#include "imex.h"

#include <map>
#include <string>
#include <vector>

namespace atomstruct {

class ATOMSTRUCT_IMEX CoordSet {
    friend class Atom;
    friend class AtomicStructure;

public:
    typedef std::vector<basegeom::Coord>  Coords;

private:
    Coords  _coords;
    int  _cs_id;
    std::map<const Atom *, float>  _bfactor_map;
    std::map<const Atom *, float>  _occupancy_map;
    AtomicStructure*  _structure;
    CoordSet(AtomicStructure* as, int cs_id);
    CoordSet(AtomicStructure* as, int cs_id, int size);

public:
    void  add_coord(const Point &coord) { _coords.push_back(coord); }
    const Coords &  coords() const { return _coords; }
    virtual  ~CoordSet();
    float  get_bfactor(const Atom *) const;
    float  get_occupancy(const Atom *) const;
    void  fill(const CoordSet *source) { _coords = source->_coords; }
    int  id() const { return _cs_id; }
    void  set_bfactor(const Atom *, float);
    void  set_occupancy(const Atom *, float);
};

}  // namespace atomstruct

#endif  // atomic_CoordSet
