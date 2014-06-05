// vim: set expandtab ts=4 sw=4:
#ifndef atomic_Bond
#define atomic_Bond

#include "base-geom/Real.h"
#include "base-geom/Connection.h"
#include "imex.h"

class Atom;
class AtomicStructure;

class ATOMSTRUCT_IMEX Bond: public Connection<Atom, Bond> {
    friend class AtomicStructure;
public:
    typedef End_points  Atoms;

private:
    Bond(AtomicStructure *, Atom *, Atom *);

public:
    const Atoms&  atoms() const { return end_points(); }
    Atom *  other_atom(Atom *a) const { return other_end(a); }
    Atom *  polymeric_start_atom() const;
};
#endif  // atomic_Bond
