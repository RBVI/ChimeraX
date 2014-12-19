// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Bond
#define atomstruct_Bond

#include <vector>

#include <basegeom/Connection.h>
#include "imex.h"

namespace atomstruct {

class Atom;
class AtomicStructure;
class Ring;

class ATOMSTRUCT_IMEX Bond: public basegeom::UniqueConnection<Atom, Bond> {
    friend class AtomicStructure;
public:
    typedef End_points  Atoms;
    typedef std::vector<const Ring*>  Rings;

private:
    Bond(AtomicStructure *, Atom *, Atom *);
    const char*  err_msg_exists() const
        { return "Bond already exists between these atoms"; }
    const char*  err_msg_loop() const
        { return "Can't bond an atom to itself"; }
    mutable Rings  _rings;

public:
    const Rings&  all_rings(bool cross_residues=false,
                                        int size_threshold=0) const;
    const Atoms&  atoms() const { return end_points(); }
    // length() inherited from UniqueConnection
    const Rings&  minimum_rings(bool cross_residues = false) const {
        return rings(cross_residues, 0);
    }
    Atom *  other_atom(Atom *a) const { return other_end(a); }
    Atom *  polymeric_start_atom() const;
    const Rings&  rings(bool cross_residues = false,
                        int all_size_threshold = 0) const;
    // sqlength() inherited from UniqueConnection
};

#include "AtomicStructure.h"
inline const Bond::Rings&
Bond::all_rings(bool cross_residues, int size_threshold) const
{
    int max_ring_size = size_threshold;
    if (max_ring_size == 0)
        max_ring_size = atoms()[0]->structure()->num_atoms();
    return rings(cross_residues, max_ring_size);
}

inline const Bond::Rings&
Bond::rings(bool cross_residues, int all_size_threshold) const
{
    atoms()[0]->structure()->rings(cross_residues, all_size_threshold);
    return _rings;
}

}  // namespace atomstruct

#endif  // atomstruct_Bond
