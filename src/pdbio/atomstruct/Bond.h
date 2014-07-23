// vim: set expandtab ts=4 sw=4:
#ifndef atomic_Bond
#define atomic_Bond

#include "basegeom/Connection.h"
#include "imex.h"

namespace atomstruct {

class Atom;
class AtomicStructure;

class ATOMSTRUCT_IMEX Bond: public basegeom::UniqueConnection<Atom, Bond> {
    friend class AtomicStructure;
public:
    typedef End_points  Atoms;

private:
    Bond(AtomicStructure *, Atom *, Atom *);
    const char*  err_msg_exists() const
        { return "Bond already exists between these atoms"; }
    const char*  err_msg_loop() const
        { return "Can't bond an atom to itself"; }

public:
    const Atoms&  atoms() const { return end_points(); }
    Atom *  other_atom(Atom *a) const { return other_end(a); }
    Atom *  polymeric_start_atom() const;
};

}  // namespace atomstruct

#endif  // atomic_Bond
