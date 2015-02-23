// vim: set expandtab ts=4 sw=4:
#ifndef atomstruct_seq_assoc
#define atomstruct_seq_assoc

#include <stdexcept>
#include <unordered_map>
#include <valarray>
#include <vector>

#include "imex.h"
#include "Sequence.h"

namespace atomstruct {

class Chain;
class Residue;

struct AssocParams {
    Sequence::Contents::size_type  est_len;
    std::vector<Sequence::Contents>  segments;
    std::vector<unsigned int>  gaps;
};

AssocParams  estimate_assoc_params(Chain&);

class ATOMSTRUCT_IMEX MatchMap {
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
private:
    std::unordered_map<SeqPos, Residue*>  _pos_to_res;
    std::unordered_map<Residue*, SeqPos>  _res_to_pos;

public:
    Sequence*  aseq;
    Chain*  mseq;
    Residue*&  operator[](SeqPos pos) { return _pos_to_res[pos]; }
    SeqPos&  operator[](Residue* r) { return _res_to_pos[r]; }
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
