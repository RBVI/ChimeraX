// vi: set expandtab ts=4 sw=4:

#include <algorithm>  // std::find_if_not, std::min
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

    // leave gaps to left and right
    std::vector<int> left_offsets, right_offsets;
    unsigned int left_errors = 0, right_offsets = 0;
    AssocParams left_ap(0, segments.begin(), segments.begin()+bsi-1,
        gaps.begin(), gaps.begin()+bsi);
    if (bsi > 0) {
        Sequence::Contents left_aseq(aseq.begin(), aseq.begin()+min_offset-2);
        left_errors = _constrained(left_aseq, left_ap, left_offsets,
            max_errors - min_errs);
    }
    AssocParams right_ap(0, segments.begin()+bsi+1, segments.end(),
        gaps.begin()+bsi+1, gaps.end());
    if (left_errors + min_errs <= max_errors && bsi+1 != segments.size()) {
        Sequence::Contents right_aseq(aseq.begin() + min_offset + longest + 1,
            aseq.end());
        right_errors = _constrained(right_aseq, right_ap, right_offsets,
            max_errors - min_errs - left_errors);
    }
    int tot_errs = min_errs + left_errors + right_errors;
    struct OffsetInfo {
        int min;
        std::vector<int> left, right;

        OffsetInfo(int m, const std::vector<int>&l, const std::vector<int>& r):
            min(m), left(l), right(r) {}
    };
    OffsetInfo offs(min_offset, left_offsets, right_offsets);

    for (int i = 0; i < err_list.size(); ++i) {
        unsigned int base_errs = err_list[i];
        if (base_errs >= std::min(tot_errs, max_errors+1))
            continue;

        int offset = left_space = i;
        if (offset == min_offset)
            continue;

        if (bsi > 0) {
            Sequence::Contents left_aseq(aseq.begin(), aseq.begin()+offset-2);
            left_errors = _constrained(left_aseq, left_ap, left_offsets,
                std::min(tot_errs, max_errors) - base_errs);
        } else {
            left_offsets.clear();
            left_errors = 0;
        }

        if (left_errors + base_errs > max_errors)
            continue;

        if (bsi+1 < segments.size()) {
            Sequence::Contents right_aseq(aseq.begin()+offset+longest+1,
                aseq.end());
            right_errors = _constrained(right_aseq, right_ap, right_offsets,
                std::min(tot_errs, max_errors) - base_errs - left_errors);
        } else {
            right_offsets.clear();
            right_errors = 0;
        }

        int err_sum = base_errs + left_errors + right_errors
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
constrained_match(const Sequence::Contents& aseq, const Chain& mseq,
    const AssocParams& ap, unsigned int max_errors)
{
    // all the segments should fit in aseq
    AssocParams cm_ap = ap;
    cm_ap.gaps.insert(cm_ap.gaps.begin(), -1);
    cm_ap.gaps.push_back(-1);
    std::vector<int> offsets;
    unsigned int errors = _constrained(aseq, cm_ap, max_errors, offsets);
    if (errors > max_errors)
        throw SA_AssocFailure("bad assoc");
    if (offsets.size() != ap.segments.size())
        throw std::logic_error("Internal match problem: #segments != #offsets");
    AssocRetvals ret;
    unsigned int res_offset = 0;
    for (int si = 0; si < ap.segments.size(); ++si) {
        int offset = offsets[si];
        Sequence::Contents& segment = ap.segments[si];
        for (int i = 0; i < segment.size(); ++i) {
            Residue* r = mseq.residues()[res_offset+i];
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
gapped_match(const Sequence::Contents& aseq, const Chain& mseq,
    const AssocParams& ap, unsigned int max_errors)
{
    Sequence::Contents gapped = ap.segments[0];
    for (int i = 0; i < ap.segments.size(); ++i) {
        gapped.insert(gapped.end(), ap.gaps[i], '.');
        gapped.insert(gapped.end(),
            ap.segments[i].begin(), ap.segments[i].end());
    }

    // to avoid matching completely in gaps, need to establish 
    // a minimum number of matches
    int min_matches = std::min(aseq.size(), mseq.size()) / 2;
    int best_score = 0, best_offset;
    unsigned int tot_errs = max_errors + 1;
    int o_end = ap.est_len - aseq.size() + 1;
    for (int offset = gapped.size() - ap.est_len; offset < o_end; ++offset) {
        int matches = 0;
        int errors = 0;
        if (offset + aseq.size() < min_matches)
            continue;
        if (gapped.size() - offset < min_matches)
            continue;
        for (int i = 0; i < aseq.size(); ++i) {
            if (offset+i < 0)
                continue;
            if (offset+i >= gapped.size())
                // in ending gap
                continue;
            auto gap_char = gapped[offset+i];
            if (aseq[i] == gap_char) {
                ++matches;
                continue;
            }
            if (gap_char == '.')
                continue;
            if (++errors >= tot_errors)
                break;
        }
        if (errors < tot_errors) {
            if (matches < min_matches || matches - errors <= best_score)
                continue;
            best_score = matches - errors;
            tot_errs = errors;
            best_offset = offset;
        }
    }

    if (tot_errs > max_errors)
        throw SA_AssocFailure("bad assoc");

    AssocRetvals ret;
    ret.num_errors = tot_errs;
    int mseq_index = 0;
    for (int i = 0; i < best_offset + aseq.size(); ++i) {
        if (i >= gapped.size())
            break;
        if (gapped[i] == '.')
            continue;
        if (i >= best_offset) {
            auto res = mseq.residues[mseq_index];
            if (res != nullptr) {
                int aseq_index = i - best_offset;
                ret.match_map[res] = aseq_index;
                ret.match_map[aseq_index] = res;
            }
        }
        ++mseq_index;
    }
    return ret;
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
