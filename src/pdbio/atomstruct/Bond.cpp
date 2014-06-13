// vim: set expandtab ts=4 sw=4:
#include "Bond.h"
#include "Atom.h"
#include <stdexcept>

namespace atomstruct {

Bond::Bond(AtomicStructure *as, Atom *a1, Atom *a2):
    Connection<Atom, Bond>(a1, a2, "Can't bond an atom to itself",
        "Bond already exists between these atoms")
{
    if (a1->structure() != as || a2->structure() != as)
        throw std::invalid_argument("Cannot bond atoms in different molecules");
}

}  // namespace atomstruct
