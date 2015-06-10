// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Bond
#define atomstruct_Bond

#include <set>
#include <vector>

#include <basegeom/Connection.h>
#include "imex.h"

namespace atomstruct {

class Atom;
class AtomicStructure;
class Residue;
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
    virtual ~Bond() {}
    const Rings&  all_rings(bool cross_residues = false, int size_threshold = 0,
        std::set<const Residue*>* ignore = nullptr) const;
    const Atoms&  atoms() const { return end_points(); }
    // length() inherited from UniqueConnection
    const Rings&  minimum_rings(bool cross_residues = false,
            std::set<const Residue*>* ignore = nullptr) const {
        return rings(cross_residues, 0, ignore);
    }
    Atom *  other_atom(Atom *a) const { return other_end(a); }
    Atom *  polymeric_start_atom() const;
    const Rings&  rings(bool cross_residues = false, int all_size_threshold = 0,
        std::set<const Residue*>* ignore = nullptr) const;
    // sqlength() inherited from UniqueConnection
};

}  // namespace atomstruct

#include "AtomicStructure.h"
inline const atomstruct::Bond::Rings&
atomstruct::Bond::all_rings(bool cross_residues, int size_threshold,
    std::set<const Residue*>* ignore) const
{
    int max_ring_size = size_threshold;
    if (max_ring_size == 0)
        max_ring_size = atoms()[0]->structure()->num_atoms();
    return rings(cross_residues, max_ring_size, ignore);
}

inline const atomstruct::Bond::Rings&
atomstruct::Bond::rings(bool cross_residues, int all_size_threshold,
    std::set<const Residue*>* ignore) const
{
    atoms()[0]->structure()->rings(cross_residues, all_size_threshold, ignore);
    return _rings;
}

#endif  // atomstruct_Bond
