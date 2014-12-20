// vi: set expandtab ts=4 sw=4:
#include "Atom.h"
#include "Bond.h"
#include "Residue.h"
#include "Sequence.h"
#include <stdexcept>

namespace atomstruct {

Bond::Bond(AtomicStructure *as, Atom *a1, Atom *a2):
    UniqueConnection<Atom, Bond>(a1, a2)
{
    if (a1->structure() != as || a2->structure() != as)
        throw std::invalid_argument("Cannot bond atoms in different molecules");
}

Atom *
Bond::polymeric_start_atom() const
{
    const Atoms& as = atoms();
    Atom *a1 = as[0];
    Atom *a2 = as[1];
    Residue *r1 = a1->residue();
    Residue *r2 = a2->residue();
    if (r1 == r2)
        return nullptr;

    unsigned char c1 = Sequence::rname3to1(r1->name());
    unsigned char c2 = Sequence::rname3to1(r2->name());
    if (c1 == 'X' || c2 == 'X')
        return nullptr;

    bool n1 = Sequence::nucleic3to1(r1->name()) != 'X';
    bool n2 = Sequence::nucleic3to1(r2->name()) != 'X';
    if (n1 != n2)
        return nullptr;

    if (n1) {
        // both nucleic
        if (a1->name() == "O3'" && a2->name() == "P")
            return a1;
        if (a1->name() == "P" && a2->name() == "O3'")
            return a2;
    } else {
        // both protein
        if (a1->name() == "C" && a2->name() == "N")
            return a1;
        if (a1->name() == "N" && a2->name() == "C")
            return a2;
    }
    return nullptr;
}

}  // namespace atomstruct
