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

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "Bond.h"
#include "Chain.h"
#include "PBGroup.h"
#include "Residue.h"
#include "Sequence.h"
#include "Structure.h"

#include <cstdlib>  // integer abs()
#include <pyinstance/PythonInstance.instantiate.h>
#include <stdexcept>

template class pyinstance::PythonInstance<atomstruct::Bond>;

namespace atomstruct {

Bond::Bond(Structure* as, Atom* a1, Atom* a2, bool bond_only): UniqueConnection(a1, a2)
{
    if (a1->structure() != as || a2->structure() != as)
        throw std::invalid_argument("Cannot bond atoms in different molecules");

    if (!bond_only) {
        as->_form_chain_check(a1, a2, this);
        a1->structure()->_structure_cats_dirty = true;
    }
    change_tracker()->add_created(a1->structure(), this);
}

bool
Bond::in_cycle() const
{
    // faster cycle checker than full ring recomputation;
    // look in breadth-first fashion off both end simultaneously
    std::vector<const Atom*> check_list1, check_list2, next_list;
    std::set<const Atom*> visited1, visited2;
    const Atom* a1 = atoms()[0];
    const Atom* a2 = atoms()[1];
    visited1.insert(a1);
    visited2.insert(a2);
    for (auto nb1: a1->neighbors()) {
        if (nb1 != a2)
            check_list1.push_back(nb1);
    }
    for (auto nb2: a2->neighbors()) {
        if (nb2 != a1)
            check_list1.push_back(nb2);
    }
    while (!check_list1.empty() && !check_list2.empty()) {
        next_list.clear();
        for (auto a: check_list1) {
            for (auto nb: a->neighbors()) {
                if (visited1.find(a) != visited1.end())
                    continue;
                if (nb == a2)
                    return true;
                next_list.push_back(nb);
            }
        }
        check_list1.swap(next_list);

        next_list.clear();
        for (auto a: check_list2) {
            for (auto nb: a->neighbors()) {
                if (visited2.find(a) != visited2.end())
                    continue;
                if (nb == a1)
                    return true;
                next_list.push_back(nb);
            }
        }
        check_list2.swap(next_list);
    }
    return false;
}

bool
Bond::is_backbone() const
{
    const Atoms& as = atoms();
    Atom *a1 = as[0];
    Atom *a2 = as[1];

    if (a1->residue() != a2->residue())
        return polymeric_start_atom() != nullptr;

    auto backbone_names = a1->residue()->ordered_min_backbone_atom_names();
    if (backbone_names == nullptr)
        return false;
    auto i1 = std::find(backbone_names->begin(), backbone_names->end(), a1->name());
    if (i1 == backbone_names->end())
        return false;
    auto i2 = std::find(backbone_names->begin(), backbone_names->end(), a2->name());
    if (i2 == backbone_names->end())
        return false;
    auto diff = i2 - i1;
    if (std::abs(diff) == 1)
        return true;
    return false;
}

enum XResType { NonPolymer, Capping, Polymer };
static XResType
_polymer_res(Residue* r, Atom* a, bool* is_nucleic)
{
    const std::set<AtomName>* min_names;
    AtomName missing_ok;
    // F86 can be polymeric and its phosphorus is named P1
    if (a->name() == "O3'" || a->name() == "P" || a->name() == "P1") {
        // nucleic
        *is_nucleic = true;
        min_names = &Residue::na_min_backbone_names;
        missing_ok = "P";
    } else if (a->name() == "C" || a->name() == "N") {
        // amino acid
        *is_nucleic = false;
        min_names = &Residue::aa_min_backbone_names;
    } else return XResType::NonPolymer;
    auto atoms_map = r->atoms_map();
    for (auto aname: *min_names) {
        auto name_atom = atoms_map.find(aname);
        if (name_atom == atoms_map.end()) {
            if (aname == missing_ok)
                continue;
            return XResType::Capping;
        }
        auto element_name = (*name_atom).second->element().name();
        if (strlen(element_name) > 1 || aname[0] != element_name[0])
            return XResType::NonPolymer;
    }
    return XResType::Polymer;
}

static Atom*
_polymeric_start_atom(Atom* a1, Atom* a2)
{
    Residue *r1 = a1->residue();
    Residue *r2 = a2->residue();
    if (r1 == r2)
        return nullptr;

    bool n1, n2;
    unsigned char c1 = Sequence::rname3to1(r1->name());
    XResType xrt1, xrt2;
    if (c1 == 'X') {
        // some heavily modified polymeric residues may not be in
        // MODRES records (or such records may be missing...)
        xrt1 = _polymer_res(r1, a1, &n1);
        if (xrt1 == XResType::NonPolymer)
            return nullptr;
    } else {
        n1 = Sequence::nucleic3to1(r1->name()) != 'X';
    }
    unsigned char c2 = Sequence::rname3to1(r2->name());
    if (c2 == 'X') {
        xrt2 = _polymer_res(r2, a2, &n2);
        if (xrt2 == XResType::NonPolymer)
            return nullptr;
    } else {
        n2 = Sequence::nucleic3to1(r2->name()) != 'X';
    }

    // are they both the same kind (amino acid / nucleic acid)?
    if (n1 != n2)
        return nullptr;

    if (c1 == 'X' && c2 == 'X' && xrt1 == XResType::Capping && xrt2 == XResType::Capping)
        return nullptr;

    if (n1) {
        // both nucleic
        // F86 can be polymeric and its phosphorus is named P1
        if (a1->name() == "O3'" && (a2->name() == "P" || a2->name() == "P1")) {
            return a1;
        }
        if ((a1->name() == "P" || a1->name() == "P1") && a2->name() == "O3'") {
            return a2;
        }
    } else {
        // both protein
        if (a1->name() == "C" && a2->name() == "N") {
            return a1;
        }
        if (a1->name() == "N" && a2->name() == "C") {
            return a2;
        }
    }
    return nullptr;
}

// polymer_bond_atoms:  if the 'first' atom (e.g. N) were bonded to the second atom
// (e.g. C), would that be a polymeric bond?
bool
Bond::polymer_bond_atoms(Atom* first, Atom* second)
{
    return _polymeric_start_atom(first, second) == first;
}

Atom*
Bond::polymeric_start_atom() const
{
    const Atoms& as = atoms();
    Atom *a1 = as[0];
    Atom *a2 = as[1];
    auto psa = _polymeric_start_atom(a1, a2);
    return psa;
}

std::vector<Atom*>
Bond::side_atoms(const Atom* side_atom) const
// all the atoms on a particular side of a bond, considering missing structure as connecting;
// raises logic_error if the other side of the bond is reached (so that Python throws ValueError)
{
    if (side_atom != _atoms[0] && side_atom != _atoms[1])
        throw std::invalid_argument("Atom given to Bond::side() not in bond!");
    auto other = other_atom(side_atom);
    return side_atom->side_atoms(other, other);
}

Atom*
Bond::smaller_side() const
// considers missing structure, throws logic_error if in a cycle
{
    auto atoms1 = side_atoms(_atoms[0]); // the compiler should perform copy elision
    auto atoms2 = side_atoms(_atoms[1]); // the compiler should perform copy elision
    if (atoms1.size() < atoms2.size())
        return _atoms[0];
    return _atoms[1];
}

}  // namespace atomstruct
