// vi: set expandtab ts=4 sw=4:
#ifndef templates_CoordSet
#define    templates_CoordSet

#include <vector>
#include "Coord.h"
#include "../imex.h"

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
