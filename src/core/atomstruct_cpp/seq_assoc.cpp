// vim: set expandtab ts=4 sw=4:

#include <algorithm>  // std::find_if_not
#include <cctype>  // std::islower
#include <stdlib.h>  // std::abs

#include "AtomicStructure.h"
#include "Chain.h"
#include "Residue.h"
#include "seq_assoc.h"

namespace atomstruct {

static AssocParams
find_gaps(Chain& chain)
{
    Residue* prev_res = nullptr;
    AssocParams ap;
    ap.est_len = 0;
    auto ri = chain.residues().begin();
    auto ci = chain.begin();
    for (unsigned int i = 0; i < chain.size(); ++i, ++ri, ++ci) {
        ap.est_len += 1;
        auto res = *ri;
        if (res == nullptr) {
            // explicit gapping
            ap.est_len = chain.size();
            ap.segments.clear();
            ap.segments.emplace_back(chain.begin(), chain.end());
            ap.gaps.clear();
            return ap;
        }
        auto c = *ci;
        int gap = 0;
        if (prev_res != nullptr) {
            auto connects = prev_res->bonds_between(res, true);
            // since unlike Chimera 1, Chimera 2 does not make
            // long "covalent" bonds across gaps, some of the
            // fancy logic used in Chimera 1 is not necessary here
            if (connects.empty()) {
                gap = res->position() - prev_res->position() - 1;
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
        if (prev_res == nullptr || gap > 0) {
            ap.segments.emplace_back();
        } 
        ap.segments.back().push_back(c);
        prev_res = res;
    }
    int front_pos = chain.residues().front()->position();
    if (ap.est_len > 0 && front_pos > 1)
        ap.est_len += front_pos - 1;
    return ap;
}

AssocParams
estimate_assoc_params(Chain& chain)
{
    AssocParams ap = find_gaps(chain);

    // Try to compensate for trailing/leading gaps by
    // looking at SEQRES records or their equivalent
    auto info = chain.structure()->input_seq_info();
    auto ii = info.find(chain.chain_id());
    if (ii != info.end() && (*ii).second.size() > ap.est_len) {
        ap.est_len = (*ii).second.size();
    }
    return ap;
}

static unsigned int
_constrained(const Sequence::Contents& aseq, AssocParams& cm_ap,
    unsigned int max_errors, std::vector<int>& offsets)
{
    // find the biggest segment, (but non-all-X segments win over
    // all-X segments)
    int longest = -1;
    int seg_num = -1;
    int bsi;
    for (auto seg: cm_ap.segments) {
        seg_num++;
        if (std::find_if_not(seg.begin(), seg.end(),
                [](char c){return c == 'X';}) == seg.end()) {
            continue;
        }
        if (seg.size() > longest) {
            bsi = seg_num;
            longest = seg.size();
        }
    }
    if (longest < 0) {
        // all segments are all-X
        seg_num = -1;
        for (auto seg: cm_ap.segments) {
            seg_num++;
            if (seg.size() > longest) {
                bsi = seg_num;
                longest = seg.size();
            }
        }
    }

    auto bsi_iter = cm_ap.segments.begin() + bsi;
    int left_space = 0;
    for (auto seg_i = cm_ap.begin(); seg_i != bsi_iter; ++seg_i) {
        left_space += seg_i->size() + 1;
    }
    int right_space = 0;
    for (auto seg_i = bsi_iter+1; seg_i != cm_ap.segments.end(); ++seg_i) {
        right_space += seg_i->size() + 1;
    }
    if (aseq.size() - left_space - right_space < longest)
        return max_errors+1;

    std::vector<unsigned int> err_list;
    auto seq = cm_ap.segments[bsi];
    int min_offset = -1;
    int min_errs = -1;
    int min_gap_errs = -1;
    unsigned int target_left_gap, target_right_gap;
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
        int errors = 0;
        for (int i = 0; i < longest; ++i) {
            if (seq[i] == aseq[offset+i])
                continue;
            if (++errors > max_errors) {
                err_list.push_back(max_errors+1);
                break;
            }

        }
        if (err_list.empty() || err_list.back() != max_errors+1) {
            err_list.push_back(errors);
            int gap_errs = 0;
            if (target_left_gap >= 0) {
                gap_errs += std::abs((offset+1) - target_left_gap);
            }
            if (target_right_gap >= 0) {
                gap_errs += std::abs(1 + (aseq.size() - (offset+longest))
                    - target_right_gap);
            }
            if (min_errs < 0 || errors < min_errs
            || (errors == min_errs && gap_errs < min_gap_errs)) {
                min_errs = errors;
                min_offset = offset;
                min_gap_errs = gap_errs;
            }
        }
    }
    if (min_offset < 0)
        return max_errors+1;
    //TODO
}

AssocRetvals
constrained_match(const Sequence::Contents& aseq, const Chain& mseq,
    const AssocParams& ap, unsigned int max_errors)
{
    // all the segments should fit in aseq
    AssocParams cm_ap = ap;
    cm_ap.gaps.insert(cm_ap.gaps.begin(), -1);
    cm_ap.gaps.push_back(-1);
    std::vector<int> offsets;
    unsigned int errors = _constrained(aseq, cm_ap, max_errors, offsets);
    //TODO
}

AssocRetvals
try_assoc(const Sequence& align_seq, const Chain& mseq,
    const AssocParams &ap, unsigned int max_errors)
{
    int lower_to_upper = 'A' - 'a';
    Sequence:::Contents aseq;
    for (auto c: align_seq.ungapped()) {
        if (std::islower(c))
            aseq.push_back(c + lower_to_upper);
        else
            aseq.push_back(c);
    }

    bool assoc_failure = false;
    AssocRetvals retvals;
    try {
        if (aseq.size() >= ap.est_len)
            // TODO
            retvals = constrained_match(aseq, mseq, ap, max_errors);
        else
            // TODO
            retvals = gapped_match(aseq, mseq, ap, max_errors);
    } catch (SA_AssocFailure) {
        assoc_failure = true;
    }

    if (!assoc_failure && retvals.num_errors == 0)
        return retvals;
    // TODO
}

}  // namespace atomstruct
