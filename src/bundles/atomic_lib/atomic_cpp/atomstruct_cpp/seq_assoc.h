// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_seq_assoc
#define atomstruct_seq_assoc

#include <map>
#include <stdexcept>
#include <valarray>
#include <vector>

#include "imex.h"
#include "StructureSeq.h"

namespace atomstruct {

class Residue;

struct AssocParams {
    Sequence::Contents::size_type  est_len;
    std::vector<Sequence::Contents>  segments;
    std::vector<int>  gaps;

    AssocParams(int el,
        std::vector<Sequence::Contents>::const_iterator s_begin,
        std::vector<Sequence::Contents>::const_iterator s_end,
        std::vector<int>::const_iterator g_begin,
        std::vector<int>::const_iterator g_end):
        est_len(el), segments(s_begin, s_end), gaps(g_begin, g_end) {}
    AssocParams(): est_len(0) {}
};

AssocParams  ATOMSTRUCT_IMEX estimate_assoc_params(StructureSeq&);

class ATOMSTRUCT_IMEX MatchMap {
public:
    typedef std::map<StructureSeq::SeqPos, Residue*>  PosToRes;
    typedef std::map<Residue*, StructureSeq::SeqPos>  ResToPos;

private:
    PosToRes  _pos_to_res;
    ResToPos  _res_to_pos;

public:
    const Sequence*  aseq;
    const StructureSeq*  mseq;
    Residue*&  operator[](StructureSeq::SeqPos pos) { return _pos_to_res[pos]; }
    StructureSeq::SeqPos&  operator[](Residue* r) { return _res_to_pos[r]; }
    const PosToRes&  pos_to_res() const { return _pos_to_res; }
    const ResToPos&  res_to_pos() const { return _res_to_pos; }
};

struct AssocRetvals {
    MatchMap  match_map;
    unsigned int  num_errors;
};

AssocRetvals  ATOMSTRUCT_IMEX try_assoc(const Sequence& aseq, const StructureSeq& mseq,
    const AssocParams& ap, unsigned int max_errors = 6);

// thrown when seq-structure association fails
class ATOMSTRUCT_IMEX SA_AssocFailure : public std::runtime_error {
public:
    SA_AssocFailure(const std::string &msg) : std::runtime_error(msg) {}
};

}  // namespace atomstruct

#endif  // atomstruct_seq_assoc
