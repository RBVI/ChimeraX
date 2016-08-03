// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_chain
#define atomstruct_chain

#include "imex.h"
#include "StructureSeq.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX Chain: public StructureSeq {
public:
    Chain(const ChainID& chain_id, Structure* as);
    virtual ~Chain();

    bool  is_chain() const { return !is_sequence(); }
};

}  // namespace atomstruct

#endif  // atomstruct_chain
