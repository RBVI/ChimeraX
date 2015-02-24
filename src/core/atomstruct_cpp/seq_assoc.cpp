// vi: set expandtab ts=4 sw=4:

#include "AtomicStructure.h"
#include "Chain.h"
#include "Residue.h"
#include "seq_assoc.h"

namespace atomstruct {

static assoc_params
find_gaps(Chain& chain)
{
    Residue* prev_res = nullptr;
    assoc_params ap;
    ap.est_len = 0;
    auto ri = chain.residues().begin();
    auto ci = chain.sequence().begin();
    for (unsigned int i = 0; i < chain.size(); ++i, ++ri, ++ci) {
        ap.est_len += 1;
        auto res = *ri;
        if (res == nullptr) {
            // explicit gapping
            ap.est_len = chain.size();
            ap.segments.clear();
            ap.segments.push_back(chain.sequence());
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
        ap.est_len += front_pos;
    return ap;
}

assoc_params
estimate_assoc_params(Chain& chain)
{
    assoc_params ap = find_gaps(chain);

    // Try to compensate for trailing/leading gaps by
    // looking at SEQRES records or their equivalent
    auto info = chain.structure()->input_seq_info();
    auto ii = info.find(chain.chain_id());
    if (ii != info.end() && (*ii).second.size() > ap.est_len) {
        ap.est_len = (*ii).second.size();
    }
    return ap;
}

}  // namespace atomstruct
