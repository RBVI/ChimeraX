// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_seq_assoc
#define atomstruct_seq_assoc

#include <map>
#include <stdexcept>
#include <valarray>
#include <vector>

#include "Chain.h"
#include "imex.h"

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

AssocParams  estimate_assoc_params(Chain&);

class ATOMSTRUCT_IMEX MatchMap {
public:
    typedef std::map<Chain::SeqPos, Residue*>  PosToRes;
    typedef std::map<Residue*, Chain::SeqPos>  ResToPos;

private:
    PosToRes  _pos_to_res;
    ResToPos  _res_to_pos;

public:
    Sequence*  aseq;
    Chain*  mseq;
    Residue*&  operator[](Chain::SeqPos pos) { return _pos_to_res[pos]; }
    Chain::SeqPos&  operator[](Residue* r) { return _res_to_pos[r]; }
    const PosToRes&  pos_to_res() const { return _pos_to_res; }
    const ResToPos&  res_to_pos() const { return _res_to_pos; }
};

struct AssocRetvals {
    MatchMap  match_map;
    unsigned int  num_errors;
};

AssocRetvals  try_assoc(const Sequence& aseq, const Chain& mseq,
    const AssocParams& ap, unsigned int max_errors = 6);

// thrown when seq-structure association fails
class ATOMSTRUCT_IMEX SA_AssocFailure : public std::runtime_error {
public:
    SA_AssocFailure(const std::string &msg) : std::runtime_error(msg) {}
};

}  // namespace atomstruct

#endif  // atomstruct_seq_assoc
