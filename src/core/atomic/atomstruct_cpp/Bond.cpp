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
#include "Atom.h"
#include "Bond.h"
#include "Chain.h"
#include "PBGroup.h"
#include "Structure.h"
#include "Residue.h"
#include "Sequence.h"
#include <stdexcept>

namespace atomstruct {

Bond::Bond(Structure* as, Atom* a1, Atom* a2): UniqueConnection(a1, a2)
{
    if (a1->structure() != as || a2->structure() != as)
        throw std::invalid_argument("Cannot bond atoms in different molecules");

    if (a1->structure()->_chains != nullptr) { // chains have been computed
        auto start_a = polymeric_start_atom();
        if (start_a != nullptr) {
            auto other_a = other_atom(start_a);
            auto start_r = start_a->residue();
            auto other_r = other_a->residue();
            if (start_r->chain() == nullptr) {
                if (other_r->chain() == nullptr) {
                    // form a new chain based on start residue's chain ID
                    auto chain = start_r->structure()->_new_chain(start_r->chain_id());
                    chain->push_back(start_r);
                    chain->push_back(other_r);
                } else {
                    // incorporate start_r into other_r's chain
                    other_r->chain()->push_front(start_r);
                }
            } else {
                if (other_r->chain() == nullptr) {
                    // incorporate other_r into start_r's chain
                    start_r->chain()->push_back(other_r);
                } else if (start_r->chain() != other_r->chain()) {
                    // merge other_r's chain into start_r's chain
                    // and demote other_r's chain to a plain sequence
                    *start_r->chain() += *other_r->chain();
                }
            }
        }
    }
    a1->structure()->_structure_cats_dirty = true;
}

enum XResType { NonPolymer, Capping, Polymer };
static XResType
_polymer_res(Residue* r, Atom* a, bool* is_nucleic)
{
    const std::set<AtomName>* min_names;
    AtomName missing_ok;
    if (a->name() == "O3'" || a->name() == "P") {
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
_polymeric_start_atom(Atom* a1, Atom* a2, Residue::PolymerType* pt = nullptr)
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
        if (pt != nullptr)
            *pt = Residue::PT_NUCLEIC;
        if (a1->name() == "O3'" && a2->name() == "P") {
            return a1;
        }
        if (a1->name() == "P" && a2->name() == "O3'") {
            return a2;
        }
    } else {
        // both protein
        if (pt != nullptr)
            *pt = Residue::PT_AMINO;
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
    Residue::PolymerType pt;
    auto psa = _polymeric_start_atom(a1, a2, &pt);
    if (psa != nullptr) {
        a1->residue()->set_polymer_type(pt);
        a2->residue()->set_polymer_type(pt);
    }
    return psa;
}

std::vector<Atom*>
Bond::side_atoms(const Atom* side_atom) const
// all the atoms on a particular side of a bond, considering missing structure as connecting;
// raises logic_error if the other side of the bond is reached (so that Python throws ValueError)
{
    if (side_atom != _atoms[0] && side_atom != _atoms[1])
        throw std::invalid_argument("Atom given to Bond::side() not in bond!");
    std::map<Atom*, std::vector<Atom*>> pb_connections;
    auto pbg = const_cast<Structure*>(_atoms[0]->structure())->pb_mgr().get_group(
        Structure::PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg != nullptr) {
        for (auto& pb: pbg->pseudobonds()) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            pb_connections[a1].push_back(a2);
            pb_connections[a2].push_back(a1);
        }
    }
    std::vector<Atom*> side_atoms;
    side_atoms.push_back(const_cast<Atom*>(side_atom));
    std::set<const Atom*> seen;
    seen.insert(side_atom);
    std::vector<Atom*> to_do;
    const Atom* other = other_atom(side_atom);
    for (auto nb: side_atom->neighbors())
        if (nb != other)
            to_do.push_back(nb);
    while (to_do.size() > 0) {
        auto a = to_do.back();
        to_do.pop_back();
        if (seen.find(a) != seen.end())
            continue;
        if (a == other)
            throw std::logic_error("Bond::side() called on bond in ring or cycle");
        seen.insert(a);
        side_atoms.push_back(a);
        if (pb_connections.find(a) != pb_connections.end()) {
            for (auto conn: pb_connections[a]) {
                to_do.push_back(conn);
            }
        }
        for (auto nb: a->neighbors())
            to_do.push_back(nb);
    }
    return side_atoms;
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
