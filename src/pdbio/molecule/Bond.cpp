// vim: set expandtab ts=4 sw=4:
#include "Bond.h"
#include "Atom.h"
#include <stdexcept>

Bond::Bond(Molecule *m, Atom *a1, Atom *a2):
    Connection<Atom, Bond>(a1, a2, "Can't bond an atom to itself",
        "Bond already exists between these atoms")
{
    if (a1->molecule() != m || a2->molecule() != m)
        throw std::invalid_argument("Cannot bond atoms in different molecules");
}
