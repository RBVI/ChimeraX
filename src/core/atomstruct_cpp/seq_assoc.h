// vim: set expandtab ts=4 sw=4:
#ifndef atomstruct_seq_assoc
#define atomstruct_seq_assoc

#include <vector>

namespace atomstruct {

class Chain;

struct assoc_params {
    int  est_len;
    std::vector<std::vector<unsigned char>>  segments;
    std::vector<unsigned int>  gaps;
};

assoc_params estimate_assoc_params(Chain&);

}  // namespace atomstruct

#endif  // atomstruct_seq_assoc
