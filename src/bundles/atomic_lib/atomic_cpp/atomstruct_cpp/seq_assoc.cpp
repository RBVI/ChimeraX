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

#include <algorithm>  // std::find_if_not, std::min
#include <cctype>  // std::islower
#include <cmath>  // std::abs

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "polymer.h"
#include "Residue.h"
#include "seq_assoc.h"
#include "Structure.h"
#include "StructureSeq.h"

namespace atomstruct {

static AssocParams
find_gaps(StructureSeq& sseq)
{
    Residue* prev_res = nullptr;
    AssocParams ap;
    ap.est_len = 0;
    auto ri = sseq.residues().begin();
    auto ci = sseq.begin();
    for (unsigned int i = 0; i < sseq.size(); ++i, ++ri, ++ci) {
        ap.est_len += 1;
        auto res = *ri;
        if (res == nullptr) {
            // explicit gapping
            ap.est_len = sseq.size();
            ap.segments.clear();
            ap.segments.emplace_back(sseq.begin(), sseq.end());
            ap.gaps.clear();
            return ap;
        }
        auto c = *ci;
        int gap = 0;
        if (prev_res != nullptr) {
            // since unlike Chimera, ChimeraX does not make
            // long "covalent" bonds across gaps, some of the
            // fancy logic used in Chimera is not necessary here
            if (!res->connects_to(prev_res)) {
                gap = res->number() - prev_res->number() - 1;
                if (gap == -1) {
                    // 1ton/1bil have gaps in insertion codes...
                    auto prev_insert = prev_res->insertion_code();
                    auto insert = res->insertion_code();
                    if (insert != ' ' && prev_insert < insert) {
                        if (prev_insert == ' ')
                            gap = insert - 'A';
                        else
                            gap = insert - prev_insert;
                    }
                }
                // CA/P-only structures have different "connectivity" criteria
                bool no_gap = false;
                bool ca_only = (res->atoms().size() == 1 && prev_res->atoms().size() == 1);
                if (gap == 0) {
                    Atom* a1;
                    Atom* a2;
                    auto pt = res->polymer_type();
                    if (ca_only) {
                        a1 = res->atoms()[0];
                        a2 = prev_res->atoms()[0];
                    } else {
                        // 2a06 has just a missing backbone nitrogen between residues 20 and 21
                        // on chains B and O; we will co-op the CA-CA criteria...
                        auto& backbone_names = (pt == PT_AMINO) ?
                            Residue::aa_min_ordered_backbone_names :
                            Residue::na_min_ordered_backbone_names;
                        for (auto i = backbone_names.rbegin(); i != backbone_names.rend(); ++i) {
                            a2 = prev_res->find_atom(*i);
                            if (a2 != nullptr)
                                break;
                        }
                        if (a2 != nullptr) {
                            for (auto bb_name: backbone_names) {
                                a1 = res->find_atom(bb_name);
                                if (a1 != nullptr) {
                                    if (a2->connects_to(a1))
                                        no_gap = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (no_gap == false) {
                        if (a2 != nullptr && a1 != nullptr) {
                            Real distsq_cutoff = pt == PT_AMINO ?
                                Residue::TRACE_PROTEIN_DISTSQ_CUTOFF :
                                Residue::TRACE_NUCLEIC_DISTSQ_CUTOFF;
                            no_gap = a1->coord().sqdistance(a2->coord()) < distsq_cutoff;
                        }
                    }
                }
                if (!no_gap) {
                    if (gap < 1)
                        // Instead of jamming everything together and hoping,
                        //   use 1 as the gap size, since the association
                        //   algorithm doesn't actually care about the size of
                        //   the gap except in tie-breaking cases (where placing
                        //   a segment in two different places would otherwise
                        //   have the same score).  Using 1 instead of jamming
                        //   allows 3oe0 chain A to work, where the gap is
                        //   between a normal-numbered segment and a long
                        //   insertion that was numbered much higher [well,
                        //   really between the insertion and the following
                        //   normal numbering]
                        gap = 1;
                    ap.gaps.push_back(gap);
                    ap.est_len += gap;
                }
            }
        }
        if (prev_res == nullptr || gap > 0) {
            ap.segments.emplace_back();
        } 
        ap.segments.back().push_back(c);
        prev_res = res;
    }
    int front_num = sseq.residues().front()->number();
    if (ap.est_len > 0 && front_num > 1)
        ap.est_len += front_num - 1;
    return ap;
}

AssocParams
estimate_assoc_params(StructureSeq& sseq)
{
    AssocParams ap = find_gaps(sseq);

    // Try to compensate for trailing/leading gaps by
    // looking at SEQRES records or their equivalent
    auto info = sseq.structure()->input_seq_info();
    auto ii = info.find(sseq.chain_id());
    if (ii != info.end() && (*ii).second.size() > ap.est_len) {
        ap.est_len = (*ii).second.size();
    }
    return ap;
}

static unsigned int
_constrained(const Sequence::Contents& aseq, AssocParams& cm_ap,
    std::vector<int>& offsets, unsigned int max_errors)
{
    // find the biggest segment, (but non-all-X/? segments win over
    // all-X/? segments)
    unsigned int longest;
    bool longest_found = false;
    int seg_num = -1;
    unsigned int bsi;
    for (auto seg: cm_ap.segments) {
        seg_num++;
        if (std::find_if_not(seg.begin(), seg.end(),
                [](char c){return c == 'X' || c == '?';}) == seg.end()) {
            continue;
        }
        if (!longest_found || seg.size() > longest) {
            bsi = seg_num;
            longest = seg.size();
            longest_found = true;
        }
    }
    if (!longest_found) {
        // all segments are all-X/?
        seg_num = -1;
        for (auto seg: cm_ap.segments) {
            seg_num++;
            if (!longest_found || seg.size() > longest) {
                bsi = seg_num;
                longest = seg.size();
                longest_found = true;
            }
        }
    }

    auto bsi_iter = cm_ap.segments.begin() + bsi;
    unsigned int left_space = 0;
    for (auto seg_i = cm_ap.segments.begin(); seg_i != bsi_iter; ++seg_i) {
        left_space += seg_i->size() + 1;
    }
    unsigned int right_space = 0;
    for (auto seg_i = bsi_iter+1; seg_i != cm_ap.segments.end(); ++seg_i) {
        right_space += seg_i->size() + 1;
    }
    if (aseq.size() - left_space - right_space < longest)
        return max_errors+1;

    std::vector<unsigned int> err_list;
    auto seq = cm_ap.segments[bsi];
    int min_offset = -1;
    unsigned int min_errs;
    bool min_errs_found = false;
    unsigned int min_gap_errs;
    int target_left_gap, target_right_gap;
    if (bsi == 0) {
        target_left_gap = cm_ap.gaps[0];
    } else {
        target_left_gap = -1;
    }
    if (bsi == cm_ap.segments.size()-1) {
        target_right_gap = cm_ap.gaps[bsi+1];
    } else {
        target_right_gap = -1;
    }
    int offset_end = aseq.size() - right_space - longest + 1;
    for (int offset = left_space; offset < offset_end; ++offset) {
        unsigned int errors = 0;
        for (unsigned int i = 0; i < longest; ++i) {
            if (seq[i] == aseq[offset+i] || seq[i] == '?')
                continue;
            if (++errors > max_errors) {
                break;
            }

        }
        err_list.push_back(errors);
        if (errors < max_errors+1) {
            unsigned int gap_errs = 0;
            if (target_left_gap >= 0) {
                gap_errs += std::abs((int)((offset+1) - target_left_gap));
            }
            if (target_right_gap >= 0) {
                gap_errs += std::abs((int)(1 + (aseq.size() - (offset+longest))
                    - target_right_gap));
            }
            if (!min_errs_found || errors < min_errs
            || (errors == min_errs && gap_errs < min_gap_errs)) {
                min_errs = errors;
                min_errs_found = true;
                min_offset = offset;
                min_gap_errs = gap_errs;
            }
        }
    }
    if (min_offset < 0)
        return max_errors+1;

    // leave gaps to left and right
    std::vector<int> left_offsets, right_offsets;
    unsigned int left_errors = 0, right_errors = 0;
    AssocParams left_ap(0, cm_ap.segments.cbegin(),
            cm_ap.segments.cbegin()+bsi,
            cm_ap.gaps.cbegin(), cm_ap.gaps.cbegin()+bsi+1);
    AssocParams right_ap(0, cm_ap.segments.begin()+bsi+1,
            cm_ap.segments.end(), cm_ap.gaps.begin()+bsi+1, cm_ap.gaps.end());
    if (bsi > 0) {
        Sequence::Contents left_aseq(aseq.begin(), aseq.begin()+min_offset-1);
        left_errors = _constrained(left_aseq, left_ap, left_offsets,
            max_errors - min_errs);
    }
    if (left_errors + min_errs <= max_errors && bsi+1 < cm_ap.segments.size()) {
        Sequence::Contents right_aseq(aseq.begin() + min_offset + longest + 1,
            aseq.end());
        right_errors = _constrained(right_aseq, right_ap, right_offsets,
            max_errors - min_errs - left_errors);
    }
    unsigned int tot_errs = min_errs + left_errors + right_errors;
    struct OffsetInfo {
        int min;
        std::vector<int> left, right;

        OffsetInfo(int m, const std::vector<int>&l, const std::vector<int>& r):
            min(m), left(l), right(r) {}
    };
    OffsetInfo offs(min_offset, left_offsets, right_offsets);

    for (unsigned int i = 0; i < err_list.size(); ++i) {
        unsigned int base_errs = err_list[i];
        if (base_errs >= std::min(tot_errs, max_errors+1))
            continue;

        int offset = left_space + i;
        if (offset == min_offset)
            continue;

        if (bsi > 0) {
            Sequence::Contents left_aseq(aseq.begin(), aseq.begin()+offset-1);
            left_errors = _constrained(left_aseq, left_ap, left_offsets,
                std::min(tot_errs, max_errors) - base_errs);
        } else {
            left_offsets.clear();
            left_errors = 0;
        }

        if (left_errors + base_errs > max_errors)
            continue;

        if (bsi+1 < cm_ap.segments.size()) {
            Sequence::Contents right_aseq(aseq.begin()+offset+longest+1, aseq.end());
            right_errors = _constrained(right_aseq, right_ap, right_offsets,
                std::min(tot_errs, max_errors) - base_errs - left_errors);
        } else {
            right_offsets.clear();
            right_errors = 0;
        }

        unsigned int err_sum = base_errs + left_errors + right_errors;
        if (err_sum < tot_errs) {
            tot_errs = err_sum;
            offs.min = offset;
            offs.left = left_offsets;
            offs.right = right_offsets;
        }
    }

    if (tot_errs > max_errors)
        return max_errors+1;

    offsets.swap(offs.left);
    offsets.push_back(offs.min);
    for (auto ro: offs.right) {
        offsets.push_back(ro + offs.min + longest + 1);
    }

    return tot_errs;
}

AssocRetvals
constrained_match(const Sequence::Contents& aseq, const StructureSeq& mseq,
    const AssocParams& ap, unsigned int max_errors)
{
    // all the segments should fit in aseq
    AssocParams cm_ap = ap;
    cm_ap.gaps.insert(cm_ap.gaps.begin(), -1);
    cm_ap.gaps.push_back(-1);
    std::vector<int> offsets;
    unsigned int errors = _constrained(aseq, cm_ap, offsets, max_errors);
    if (errors > max_errors)
        throw SA_AssocFailure("bad assoc");
    if (offsets.size() != ap.segments.size())
        throw std::logic_error("Internal match problem: #segments != #offsets");
    AssocRetvals ret;
    unsigned int res_offset = 0;
    auto num_mseq_res = mseq.residues().size();
    for (unsigned int si = 0; si < ap.segments.size(); ++si) {
        int offset = offsets[si];
        const Sequence::Contents& segment = ap.segments[si];
        for (unsigned int i = 0; i < segment.size(); ++i) {
            Residue* r = mseq.residues()[res_offset+i];
            if (res_offset+i >= num_mseq_res)
                break;
            if (r != nullptr) {
                ret.match_map[r] = offset+i;
                ret.match_map[offset+i] = r;
            }
        }
        res_offset += segment.size();
    }
    ret.num_errors = errors;
    return ret;
}

AssocRetvals
gapped_match(const Sequence::Contents& aseq, const StructureSeq& mseq,
    const AssocParams& ap, unsigned int max_errors)
{
    Sequence::Contents gapped = ap.segments[0];
    for (unsigned int i = 1; i < ap.segments.size(); ++i) {
        gapped.insert(gapped.end(), ap.gaps[i-1], '.');
        gapped.insert(gapped.end(),
            ap.segments[i].begin(), ap.segments[i].end());
    }

    // to avoid matching completely in gaps, need to establish 
    // a minimum number of matches
    unsigned int min_matches = std::min(aseq.size(), mseq.size()) / 2;
    int best_score = 0, best_offset;
    unsigned int tot_errs = max_errors + 1;
    int o_end = ap.est_len - aseq.size() + 1;
    for (int offset = gapped.size() - ap.est_len; offset < o_end; ++offset) {
        unsigned int matches = 0;
        unsigned int errors = 0;
        if (offset + aseq.size() < min_matches)
            continue;
        if (gapped.size() - offset < min_matches)
            continue;
        for (unsigned int i = 0; i < aseq.size(); ++i) {
            if (offset+(int)i < 0)
                continue;
            // since we know the sum is positive, the below will work
            // since the int is promoted to unsigned, and the addition
            // of two unsigns uses modulo arithmetic
            unsigned int cur_offset = offset + i;
            if (cur_offset >= gapped.size())
                // in ending gap
                continue;
            auto gap_char = gapped[cur_offset];
            if (aseq[i] == gap_char) {
                ++matches;
                continue;
            }
            if (gap_char == '.' || gap_char == '?')
                continue;
            if (++errors >= tot_errs)
                break;
        }
        if (errors < tot_errs) {
            if (matches < min_matches
            || (int)matches - (int)errors <= best_score)
                continue;
            best_score = (int)matches - (int)errors;
            tot_errs = errors;
            best_offset = offset;
        }
    }

    if (tot_errs > max_errors)
        throw SA_AssocFailure("bad assoc");

    AssocRetvals ret;
    ret.num_errors = tot_errs;
    int num_mseq_res = mseq.residues().size();
    int mseq_index = 0;
    for (int i = 0; i < best_offset + (int)aseq.size(); ++i) {
        if (i >= (int)gapped.size())
            break;
        if (gapped[i] == '.')
            continue;
        if (i >= best_offset) {
            auto res = mseq.residues()[mseq_index];
            if (res != nullptr) {
                int aseq_index = i - best_offset;
                ret.match_map[res] = aseq_index;
                ret.match_map[aseq_index] = res;
            }
        }
        ++mseq_index;
        if (mseq_index >= num_mseq_res)
            break;
    }
    return ret;
}

AssocRetvals
try_assoc(const Sequence& align_seq, const StructureSeq& mseq,
    const AssocParams &ap, unsigned int max_errors)
{
    int lower_to_upper = 'A' - 'a';
    Sequence::Contents aseq;
    for (auto c: align_seq.ungapped()) {
        if (std::islower(c)) {
            aseq.push_back(c + lower_to_upper);
        } else
            aseq.push_back(c);
    }

    bool assoc_failure = false;
    AssocRetvals retvals;
    try {
        if (aseq.size() >= ap.est_len)
            retvals = constrained_match(aseq, mseq, ap, max_errors);
        else
            retvals = gapped_match(aseq, mseq, ap, max_errors);
    } catch (SA_AssocFailure&) {
        assoc_failure = true;
    }
    retvals.match_map.aseq = &align_seq;
    retvals.match_map.mseq = &mseq;

    if (!assoc_failure && retvals.num_errors == 0)
        return retvals;

    // prune off 'X/?' residues and see how that works...
    if (mseq.front() != 'X' && mseq.back() != 'X' && mseq.front() != '?' && mseq.back() != '?') {
        if (assoc_failure)
            throw SA_AssocFailure("bad assoc");
        return retvals;
    }
    StructureSeq no_X = mseq;
    auto no_X_ap = AssocParams(ap.est_len, ap.segments.begin(),
        ap.segments.end(), ap.gaps.begin(), ap.gaps.end());
    while (no_X.size() > 0 && (no_X.front() == 'X' || no_X.front() == '?')) {
        no_X.pop_front();
        --no_X_ap.est_len;
    }
    if (no_X.size() == 0) {
        if (assoc_failure)
            throw SA_AssocFailure("bad assoc");
        return retvals;
    }
    unsigned int offset = mseq.size() - no_X.size();
    while (offset > 0 && offset >= no_X_ap.segments.front().size()) {
        offset -= no_X_ap.segments.front().size();
        no_X_ap.segments.erase(no_X_ap.segments.begin());
        no_X_ap.est_len -= no_X_ap.gaps.front();
        no_X_ap.gaps.erase(no_X_ap.gaps.begin());
    }
    while (offset-- > 0)
        no_X_ap.segments.front().erase(no_X_ap.segments.front().begin());
    unsigned int tail_loss = 0;
    while (no_X.back() == 'X' || no_X.back() == '?') {
        no_X.pop_back();
        --no_X_ap.est_len;
        ++tail_loss;
    }
    while (tail_loss > 0 && tail_loss >= no_X_ap.segments.back().size()) {
        tail_loss -= no_X_ap.segments.back().size();
        no_X_ap.segments.pop_back();
        no_X_ap.est_len -= no_X_ap.gaps.back();
        no_X_ap.gaps.pop_back();
    }
    while (tail_loss-- > 0)
        no_X_ap.segments.back().pop_back();
    AssocRetvals no_X_retvals;
    try {
        if (aseq.size() >= no_X_ap.est_len)
            no_X_retvals = constrained_match(aseq, no_X, no_X_ap, max_errors);
        else
            no_X_retvals = gapped_match(aseq, no_X, no_X_ap, max_errors);
    } catch (SA_AssocFailure&) {
        if (assoc_failure)
            throw;
        return retvals;
    }
    if (assoc_failure || no_X_retvals.num_errors < retvals.num_errors)
        return no_X_retvals;
    return retvals;
}

}  // namespace atomstruct
