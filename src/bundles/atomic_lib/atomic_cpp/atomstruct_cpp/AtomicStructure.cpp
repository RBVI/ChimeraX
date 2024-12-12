// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "AtomicStructure.h"
#include "Bond.h"
#include "CoordSet.h"
#include "destruct.h"
#include "PBGroup.h"
#include "polymer.h"
#include "Pseudobond.h"
#include "Residue.h"
#include "seq_assoc.h"

#include <logger/logger.h>
#include <pysupport/convert.h>

#include <algorithm>  // for std::find, std::sort, std::remove_if, std::min
#include <cmath> // std::abs
#include <iterator>
#include <map>
#include "Python.h"
#include <stdexcept>
#include <set>
#include <type_traits> // for std::remove_reference

namespace atomstruct {

AtomicStructure *AtomicStructure::copy() const
{
    AtomicStructure *m = new AtomicStructure(_logger);
    _copy(m);
    return m;
}

void
AtomicStructure::_compute_structure_cats() const
{
    std::vector<std::vector<Atom*>> bonded;
    bonded_groups(&bonded, true);
    std::map<Atom*, std::vector<Atom*>*> group_lookup;
    std::map<Atom*, Atom*> atom_to_root;
    for (auto& grp: bonded) {
        auto root = grp[0];
        group_lookup[root] = &grp;
        for (auto a: grp)
            atom_to_root[a] = root;
    }

    //segregate into small solvents / other
    std::vector<Atom*> small_solvents;
    std::set<Atom*> root_set;
    for (auto root_grp: group_lookup) {
        auto root = root_grp.first;
        auto grp = root_grp.second;
        if (grp->size() < 4 && Residue::std_solvent_names.find(root->residue()->name())
        != Residue::std_solvent_names.end())
            small_solvents.push_back(root);
        else if (grp->size() == 1 && root->residue()->atoms().size() == 1
        && root->element().number() > 4 && root->element().number() < 9)
            small_solvents.push_back(root);
        else
            root_set.insert(root);
    }

    // determine/assign solvent
    std::map<std::string, std::vector<Atom*>> solvents;
    solvents["small solvents"] = small_solvents;
    for (auto root: root_set) {
        auto grp_size = group_lookup[root]->size();
        if (grp_size > 10)
            continue;
        if (grp_size != root->residue()->atoms().size())
            continue;

        // potential solvent
        solvents[root->residue()->name().c_str()].push_back(root);
    }
    std::string best_solvent_name;
    size_t best_solvent_size = 10;
    for (auto& sn_roots: solvents) {
        auto sn = sn_roots.first;
        auto& roots = sn_roots.second;
        if (roots.size() < best_solvent_size)
            continue;
        best_solvent_name = sn;
        best_solvent_size = roots.size();
    }
    for (auto root: small_solvents)
        for (auto a: *(group_lookup[root]))
            a->_set_structure_category(Atom::StructCat::Solvent);
    if (!best_solvent_name.empty() && best_solvent_name != "small solvents") {
        for (auto root: solvents[best_solvent_name]) {
            root_set.erase(root);
            for (auto a: *(group_lookup[root]))
                a->_set_structure_category(Atom::StructCat::Solvent);
        }
    }

    // assign ions
    std::set<Atom*> ions;
    for (auto root: root_set) {
        if (group_lookup[root]->size() == 1) {
            if (root->element().number() > 1 && !root->element().is_noble_gas())
                ions.insert(root);
        }
            
    }
    // possibly expand ion to remainder of residue (coordination complex)
    //
    // in case a large all-atom one-residue non-structure leaks into here,
    // skip if no bonds
    if (num_bonds() > 0) {
        std::set<Residue*> checked_residues;
        auto ions_copy = ions;
        for (auto root: ions_copy) {
            if (group_lookup[root]->size() == root->residue()->atoms().size())
                continue;
            if (checked_residues.find(root->residue()) != checked_residues.end())
                continue;
            checked_residues.insert(root->residue());
            std::set<Atom*> seen_roots = { root };
            for (auto a: root->residue()->atoms()) {
                auto rt = atom_to_root[a];
                if (seen_roots.find(rt) != seen_roots.end())
                    continue;
                seen_roots.insert(rt);
            }
            // add segments of less than 5 heavy atoms
            for (auto rt: seen_roots) {
                if (ions.find(rt) != ions.end())
                    continue;
                int num_heavys = 0;
                for (auto a: *(group_lookup[rt])) {
                    if (a->element().number() > 1) {
                        ++num_heavys;
                        if (num_heavys > 4)
                            break;
                    }
                }
                if (num_heavys < 5)
                    ions.insert(rt);
            }
        }
    }
    for (auto root: ions) {
        root_set.erase(root);
        for (auto a: *(group_lookup[root]))
            a->_set_structure_category(Atom::StructCat::Ions);
    }

    if (root_set.empty()) {
        _structure_cats_dirty = false;
        return;
    }

    // assign ligand

    // find longest chain
    std::vector<Atom*>* longest = nullptr;
    for (auto root: root_set) {
        auto grp = group_lookup[root];
        if (longest == nullptr || grp->size() > longest->size())
            longest = grp;
    }
    
    std::vector<Atom*> ligands;
    // CDL has 256 atoms
    auto ligand_cutoff = std::min(longest->size()/4, (size_t)256);
    for (auto root: root_set) {
        auto grp = group_lookup[root];
        if (grp->size() <= ligand_cutoff) {
            // fewer than 10 residues?
            std::set<Residue*> residues;
            for (auto a: *grp) {
                residues.insert(a->residue());
            }
            if (residues.size() < 10) {
                // ensure it isn't part of a longer chain,
                // some of which is missing...
                bool long_chain = true;
                if (root->residue()->chain() == nullptr)
                    long_chain = false;
                else if (root->residue()->chain()->residues().size() < 10)
                    long_chain = false;
                if (!long_chain)
                    ligands.push_back(root);
            }
        }
    }
    for (auto root: ligands) {
        root_set.erase(root);
        for (auto a: *(group_lookup[root]))
            a->_set_structure_category(Atom::StructCat::Ligand);
    }

    // remainder in "main" category
    for (auto root: root_set) {
        std::set<Residue*> root_residues;
        auto grp = group_lookup[root];
        for (auto a: *grp) {
            a->_set_structure_category(Atom::StructCat::Main);
            root_residues.insert(a->residue());
        }
        // try to reclassify bound ligands as ligand
        std::set<Chain*> root_chains;
        for (auto r: root_residues)
            if (r->chain() != nullptr)
                root_chains.insert(r->chain());
        std::set<Residue*> seq_residues;
        for (auto chain: root_chains) {
            for (auto r: chain->residues()) {
                if (r != nullptr)
                    seq_residues.insert(r);
            }
        }
        if (seq_residues.empty())
            continue;
        std::vector<Residue*> bound;
        std::set_difference(root_residues.begin(), root_residues.end(),
            seq_residues.begin(), seq_residues.end(), std::inserter(bound, bound.end()));
        for (auto br: bound) {
            for (auto a: br->atoms())
                a->_set_structure_category(Atom::StructCat::Ligand);
        }
    }
    _structure_cats_dirty = false;
}

void
AtomicStructure::normalize_ss_ids()
{
    for (auto& chain: chains()) {
        int h_id = 0, s_id = 0;
        Residue::SSType cur_type = Residue::SS_COIL;
        int raw_id;
        for (auto r: chain->residues()) {
            if (r == nullptr)
                continue;
            if (r->ss_type() != cur_type) {
                cur_type = r->ss_type();
                if (cur_type == Residue::SS_HELIX) {
                    raw_id = r->_ss_id;
                    r->_ss_id = ++h_id;
                } else if (cur_type == Residue::SS_STRAND) {
                    raw_id = r->_ss_id;
                    r->_ss_id = ++s_id;
                } else {
                    r->_ss_id = 0;
                }
            } else if (r->ss_type() == Residue::SS_COIL) {
                r->_ss_id = 0;
            } else if (r->_ss_id != raw_id) {
                if (cur_type == Residue::SS_HELIX) {
                    raw_id = r->_ss_id;
                    r->_ss_id = ++h_id;
                } else if (cur_type == Residue::SS_STRAND) {
                    raw_id = r->_ss_id;
                    r->_ss_id = ++s_id;
                }
            } else if (r->ss_type() == Residue::SS_HELIX) {
                r->_ss_id = h_id;
            } else {
                r->_ss_id = s_id;
            }
        }
    }
    ss_ids_normalized = true;
}

void
AtomicStructure::_make_chains() const
{
    std::set<ChainID> pre_existing;
    if (_chains == nullptr) {
        _chains = new Chains();
    } else {
        for (auto chain: *_chains)
            pre_existing.insert(chain->chain_id());
    }
    _chains_made = true; // prevent recursion

    auto polys = polymers();

    // In an ideal world there would be a one-to-one correspondence between
    // polymers and chains.  In the real world, we need to form chains
    // based on chain ID, regardless of how many polymers are involved.
    // For instance, chain A in 2duf is two polyers connected by a non-polymeric
    // residue (the chromaphore).
    typedef std::pair<ChainID,PolymerType> PolyKey;
    std::map<PolyKey,std::vector<Chain::Residues>> id_to_polys;
    for (auto polymer_type: polys) {
        auto& polymer = polymer_type.first;
        id_to_polys[PolyKey(polymer[0]->chain_id(),polymer_type.second)].push_back(polymer);
    }

    std::map<Residue*, int> res_to_index; // fill in on demand
    for (auto key_polys: id_to_polys) {
        auto id_type = key_polys.first;
        auto chain_id = id_type.first;
        if (pre_existing.find(chain_id) != pre_existing.end())
            continue;
        auto pt_type = id_type.second;
        auto& chain_polys = key_polys.second;
        std::vector<Residue*> res_list;
        for (std::remove_reference<decltype(chain_polys)>::type::size_type i = 0;
        i < chain_polys.size(); ++i) {
            auto& polymer = chain_polys[i];
            res_list.insert(res_list.end(), polymer.begin(), polymer.end());
            if (chain_polys.size() > i+1) {
                // possibly add in the residues between the polymers
                if (res_to_index.size() == 0) {
                    // fill in res_to_index
                    auto ri = residues().begin();
                    int index = 0;
                    auto end = residues().end();
                    while (ri != end)
                        res_to_index[*ri++] = index++;
                }
                auto last_res = polymer.back();
                auto start_next_poly = chain_polys[i+1].front();
                auto next_res_i = res_to_index[last_res] + 1;
                auto next_res = residues()[next_res_i];
                while (next_res != start_next_poly) {
                    if (last_res->connects_to(next_res) && next_res->chain_id() == chain_id)
                        res_list.push_back(next_res);
                    else
                        break;
                    last_res = next_res;
                    next_res = residues()[++next_res_i];
                }
            }
        }
        auto chain = _new_chain(chain_id, pt_type);

        // first, create chain directly from structure
        chain->bulk_set(res_list, nullptr);

        auto three_let_i = _input_seq_info.find(chain_id);
        if (three_let_i != _input_seq_info.end()) {
            // try to adjust chain based on SEQRES
            auto& three_let_seq = three_let_i->second;
            auto seqres_size = three_let_seq.size();
            auto chain_size = chain->size();
            if (seqres_size == chain_size) {
                // presumably no adjustment necessary
                chain->set_from_seqres(true);
                continue;
            }

            if (seqres_size < chain_size) {
                logger::warning(_logger, input_seq_source, " for chain ",
                    chain_id, " is incomplete.  "
                    "Ignoring input sequence records as basis for sequence.");
                continue;
            }

            // skip if standard residues have been removed but the
            // sequence records haven't been...
            Sequence sr_seq(three_let_seq);
            if ((unsigned)std::count(chain->begin(), chain->end(), '?') == chain_size
            && std::search(sr_seq.begin(), sr_seq.end(),
            chain->begin(), chain->end()) == sr_seq.end()) {
                logger::warning(_logger, "Residues corresponding to ",
                    input_seq_source, " for chain ", chain_id,
                    " are missing.  Ignoring record as basis for sequence.");
                continue;
            }

            // okay, seriously try to match up with SEQRES
            auto ap = estimate_assoc_params(*chain);

            // UNK residues may be jammed up against the regular sequnce
            // in SEQRES records (3dh4, 4gns) despite missing intervening
            // residues; compensate...
            //
            // can't just test against est_len since there can be other
            // missing structure

            // leading Xs...
            unsigned int additional_Xs = 0;
            unsigned int existing_Xs = 0;
            auto gi = ap.gaps.begin();
            for (auto si = ap.segments.begin(); si != ap.segments.end()
            && si+1 != ap.segments.end(); ++si, ++gi) {
                auto seg = *si;
                if (std::find_if_not(seg.begin(), seg.end(),
                [](char c){return c == '?';}) == seg.end()) {
                    // all 'X'
                    existing_Xs += seg.size();
                    additional_Xs += *gi;
                } else {
                    break;
                }
            }
            if (existing_Xs && sr_seq.size() >= existing_Xs
            && std::count(sr_seq.begin(), sr_seq.begin() + existing_Xs, '?')
            == existing_Xs)
                sr_seq.insert(sr_seq.begin(), additional_Xs, '?');

            // trailing Xs...
            additional_Xs = 0;
            existing_Xs = 0;
            auto rgi = ap.gaps.rbegin();
            for (auto rsi = ap.segments.rbegin(); rsi != ap.segments.rend()
            && rsi+1 != ap.segments.rend(); ++rsi, ++rgi) {
                auto seg = *rsi;
                if (std::find_if_not(seg.begin(), seg.end(),
                [](char c){return c == '?';}) == seg.end()) {
                    // all 'X'
                    existing_Xs += seg.size();
                    additional_Xs += *rgi;
                } else {
                    break;
                }
            }
            if (existing_Xs && sr_seq.size() >= existing_Xs
            && std::count(sr_seq.rbegin(), sr_seq.rbegin() + existing_Xs, '?')
            == existing_Xs)
                sr_seq.insert(sr_seq.end(), additional_Xs, '?');

            // if a jump in numbering is in an unresolved part of the structure,
            // the estimated length can be too long...
            if (ap.est_len > sr_seq.size())
                ap.est_len = sr_seq.size();

            // since gapping a structure sequence is considered an "error",
            // need to allow a lot more errors than normal.  However, allowing
            // a _lot_ of errors can make it take a very long time to find the
            // answer, so limit the maximum...
            // (1vqn, chain 0 is > 2700 residues)
            unsigned int seq_len = chain->size();
            unsigned int gap_sum = 0;
            for (auto gap: ap.gaps) {
                gap_sum += gap;
            }
            unsigned int max_errs = std::min(seq_len/2, std::max(seq_len/10, gap_sum));
            AssocRetvals retvals;
            try {
                retvals = try_assoc(sr_seq, *chain, ap, max_errs);
            } catch (SA_AssocFailure&) {
                chain->set_from_seqres(false);
                continue;
            }
            chain->set_from_seqres(true);
            auto& p2r = retvals.match_map.pos_to_res();
            Chain::Residues new_residues;
            for (Chain::SeqPos i = 0; i < sr_seq.size(); ++i ) {
                auto pi = p2r.find(i);
                if (pi == p2r.end())
                    new_residues.push_back(nullptr);
                else
                    new_residues.push_back((*pi).second);
            }
            chain->bulk_set(new_residues, &sr_seq.contents());
        }
    }


    // now look through missing-structure pseudobonds for "chain trace"
    // pseudobonds and set their shown_when_atoms_hidden to false, so 
    // that they disappear when ribbons/nucleotides are shown
    auto pbg = pb_mgr().get_group(PBG_MISSING_STRUCTURE);
    if (pbg != nullptr) {
        for (auto pb: pbg->pseudobonds()) {
            Residue* r1 = pb->atoms()[0]->residue();
            Residue* r2 = pb->atoms()[1]->residue();
            if (r1->chain() == nullptr || (r1->chain() != r2->chain()))
                continue;
            auto& res_map = r1->chain()->res_map();
            if (std::abs((int)(res_map.at(r1) - res_map.at(r2))) < 2) {
                Real cutoff = r1->polymer_type() == PT_AMINO ?
                    Residue::TRACE_PROTEIN_DISTSQ_CUTOFF : Residue::TRACE_NUCLEIC_DISTSQ_CUTOFF;
                if (r1->chain()->from_seqres()) {
                    // Sometimes SEQRES only contain the existing residues despite
                    // truely missing residues, so do some additional checking
                    if (pb->sqlength() <= 4 * cutoff)
                        // Okay, willing to trust that these are truly consecutive residues...
                        pb->set_shown_when_atoms_hidden(false);
                } else {
                    // need to check more closely
                    if (std::abs(r2->number() - r1->number()) < 2 && pb->sqlength() <= cutoff)
                        pb->set_shown_when_atoms_hidden(false);
                }
            }
        }
    }
}

std::vector<std::pair<Chain::Residues,PolymerType>>
AtomicStructure::polymers(AtomicStructure::PolymerMissingStructure missing_structure_treatment,
    bool consider_chain_ids) const
{
    // if consider_missing_structure is false, just consider actual
    // existing polymeric bonds (not missing-segment pseudobonds);
    // if consider_chain_ids is true, don't have a polymer span
    // a change in chain ID

    // connected polymeric residues have to be adjacent in the residue list,
    // so make an index map
    int i = 0;
    std::map<const Residue*, int> res_lookup;
    for (auto r: _residues) {
        res_lookup[r] = i++;
    }

    // Find all polymeric connections and make a map
    // keyed on residue with value of whether that residue
    // is connected to the next one
    std::map<Residue*, bool> connected;
    for (auto b: bonds()) {
        Atom* start = b->polymeric_start_atom();
        if (start != nullptr) {
            Residue* sr = start->residue();
            Residue* nr = b->other_atom(start)->residue();
            if (res_lookup[sr] + 1 == res_lookup[nr]
            && (!consider_chain_ids || sr->chain_id() == nr->chain_id()))
                // If consider_chain_ids is true,
                // if an artificial linker is used to join
                // otherwise unconnected amino acid chains,
                // they all can have different chain IDs,
                // and should be treated as separate chains (2atp)
                connected[sr] = true;
        }
    }

    if (missing_structure_treatment != PMS_NEVER_CONNECTS) {
        // go through missing-structure pseudobonds
        auto pbg = const_cast<AtomicStructure*>(this)->_pb_mgr.get_group(
            PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
        if (pbg != nullptr) {
            for (auto& pb: pbg->pseudobonds()) {
                Atom* a1 = pb->atoms()[0];
                Atom* a2 = pb->atoms()[1];
                Residue *r1 = a1->residue();
                Residue *r2 = a2->residue();
                if (missing_structure_treatment == PMS_TRACE_CONNECTS) {
                    if (std::abs(r1->number() - r2->number()) > 1)
                        continue;
                    Atom* pa1 = r1->principal_atom();
                    if (r1->principal_atom() == nullptr)
                        continue;
                    Atom* pa2 = r2->principal_atom();
                    if (r2->principal_atom() == nullptr)
                        continue;
                    Real distsq_cutoff;
                    bool protein = pa1->name() == "CA";
                    if (protein) {
                        distsq_cutoff = Residue::TRACE_PROTEIN_DISTSQ_CUTOFF;
                    } else {
                        distsq_cutoff = Residue::TRACE_NUCLEIC_DISTSQ_CUTOFF;
                    }
                    if (pa1->coord().sqdistance(pa2->coord()) > distsq_cutoff)
                        continue;
                }
                int index1 = res_lookup[r1], index2 = res_lookup[r2];
                if (abs(index1 - index2) == 1
                && r1->chain_id() == r2->chain_id()) {
                    if (index1 < index2) {
                        connected[r1] = true;
                    } else {
                        connected[r2] = true;
                    }
                }
            }
        }
    }

    // Go through residue list; start chains with initially-connected residues
    std::vector<std::pair<Chain::Residues,PolymerType>> polys;
    Chain::Residues chain;
    bool in_chain = false;
    PolymerType pt = PT_NONE;
    for (auto& upr: _residues) {
        Residue* r = upr;
        auto connection = connected.find(r);
        if (connection == connected.end()) {
            if (in_chain) {
                chain.push_back(r);
                if (pt == PT_NONE) {
                    // all 'X'; look at residue
                    pt = r->find_atom("CA") ? PT_AMINO : PT_NUCLEIC;
                }
                polys.emplace_back(chain, pt);
                chain.clear();
                in_chain = false;
                pt = PT_NONE;
            }
        } else {
            chain.push_back(r);
            in_chain = true;
            if (pt == PT_NONE)
                pt = Sequence::rname_polymer_type(r->name());
        }
    }
    if (in_chain) {
        if (pt == PT_NONE) {
            // all 'X'; look at residue
            pt = chain.back()->find_atom("CA") ? PT_AMINO : PT_NUCLEIC;
        }
        polys.emplace_back(chain, pt);
    }

    _polymers_computed = true;
    return polys;
}

}  // namespace atomstruct
