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

#ifndef atomstruct_CoordSet
#define atomstruct_CoordSet

#include <pyinstance/PythonInstance.declare.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "Coord.h"
#include "imex.h"
#include "session.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX CoordSet: public pyinstance::PythonInstance<CoordSet> {
    friend class Atom;
    friend class Structure;

public:
    typedef std::vector<Coord>  Coords;

private:
    Coords  _coords;
    int  _cs_id;
    std::unordered_map<const Atom *, float>  _bfactor_map;
    std::unordered_map<const Atom *, float>  _occupancy_map;
    Structure*  _structure;
    CoordSet(Structure* as, int cs_id);
    CoordSet(Structure* as, int cs_id, int size);

public:
    CoordSet& operator=(const CoordSet& source) {
        if (this != &source) {
            _coords = source._coords;
            _bfactor_map = source._bfactor_map; _occupancy_map = source._occupancy_map;
        }
        return *this;
    }
    void  add_coord(const Point &coord) { _coords.push_back(coord); }
    const Coords &  coords() const { return _coords; }
    void set_coords(Real *xyz, size_t n);
    virtual  ~CoordSet();
    float  get_bfactor(const Atom *) const;
    float  get_occupancy(const Atom *) const;
    void  fill(const CoordSet *source) { _coords = source->_coords; }
    int  id() const { return _cs_id; }
    int  session_num_floats(int /*version*/=CURRENT_SESSION_VERSION) const {
        return _bfactor_map.size() + _occupancy_map.size() + 3 * _coords.size();
    }
    int  session_num_ints(int /*version*/=CURRENT_SESSION_VERSION) const {
        return _bfactor_map.size() + _occupancy_map.size() + 3;
    }
    void  session_restore(int version, int** ints, float** floats);
    void  session_save(int** ints, float** floats) const;
    void  set_bfactor(const Atom* a, float val) { _bfactor_map[a] = val; }
    void  set_occupancy(const Atom* a, float val) { _occupancy_map[a] = val; }
    Structure*  structure() const { return _structure; }
};

}  // namespace atomstruct

#endif  // atomstruct_CoordSet
