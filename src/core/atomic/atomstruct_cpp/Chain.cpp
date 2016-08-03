// vi: set expandtab ts=4 sw=4:

#define ATOMSTRUCT_EXPORT
#include "Chain.h"
#include "ChangeTracker.h"
#include "destruct.h"
#include "Structure.h"

namespace atomstruct {

Chain::Chain(const ChainID& chain_id, Structure* s): StructureSeq(chain_id, s)
{
    if (is_chain())
        _structure->change_tracker()->add_created(this);
}

Chain::~Chain()
{
    if (is_chain())
        _structure->change_tracker()->add_deleted(this);
}
}  // namespace atomstruct
