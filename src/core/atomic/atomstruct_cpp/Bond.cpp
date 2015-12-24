// vi: set expandtab ts=4 sw=4:
#include "Atom.h"
#include "Bond.h"
#include "Chain.h"
#include "Residue.h"
#include "Sequence.h"
#include <stdexcept>

namespace atomstruct {

Bond::Bond(AtomicStructure *as, Atom *a1, Atom *a2): UniqueConnection<Atom>(a1, a2)
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

static bool
_polymer_res(Residue* r, Atom* a, bool* is_nucleic)
{
    const std::set<AtomName>* min_names;
    AtomName missing_ok;
    if (a->name() == "O3'" || a->name() == "P") {
        // nucleic
        *is_nucleic = true;
        min_names = &Residue::na_min_backbone_names;
        missing_ok = "P";
    } else {
        // amino acid
        *is_nucleic = false;
        min_names = &Residue::aa_min_backbone_names;
    }
    auto atoms_map = r->atoms_map();
    for (auto aname: *min_names) {
        auto name_atom = atoms_map.find(aname);
        if (name_atom == atoms_map.end()) {
            if (aname == missing_ok)
                continue;
            return false;
        }
        auto element_name = (*name_atom).second->element().name();
        if (strlen(element_name) > 1 || aname[0] != element_name[0])
            return false;
    }
    return true;
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

    bool n1, n2;
    unsigned char c1 = Sequence::rname3to1(r1->name());
    if (c1 == 'X') {
        // some heavily modified polymeric residues may not be in
        // MODRES records (or such records may be missing...)
        if (!_polymer_res(r1, a1, &n1))
            return nullptr;
    } else {
        n1 = Sequence::nucleic3to1(r1->name()) != 'X';
    }
    unsigned char c2 = Sequence::rname3to1(r2->name());
    if (c2 == 'X') {
        if (!_polymer_res(r2, a2, &n2))
            return nullptr;
    } else {
        n2 = Sequence::nucleic3to1(r2->name()) != 'X';
    }

    // are they both the same kind (amino acid / nucleic acid)?
    if (n1 != n2)
        return nullptr;

    if (n1) {
        // both nucleic
        if (a1->name() == "O3'" && a2->name() == "P") {
            r1->set_polymer_type(Residue::PT_NUCLEIC);
            r2->set_polymer_type(Residue::PT_NUCLEIC);
            return a1;
        }
        if (a1->name() == "P" && a2->name() == "O3'") {
            r1->set_polymer_type(Residue::PT_NUCLEIC);
            r2->set_polymer_type(Residue::PT_NUCLEIC);
            return a2;
        }
    } else {
        // both protein
        if (a1->name() == "C" && a2->name() == "N") {
            r1->set_polymer_type(Residue::PT_AMINO);
            r2->set_polymer_type(Residue::PT_AMINO);
            return a1;
        }
        if (a1->name() == "N" && a2->name() == "C") {
            r1->set_polymer_type(Residue::PT_AMINO);
            r2->set_polymer_type(Residue::PT_AMINO);
            return a2;
        }
    }
    return nullptr;
}

}  // namespace atomstruct
