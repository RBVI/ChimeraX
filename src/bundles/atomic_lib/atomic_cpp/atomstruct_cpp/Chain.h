// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_chain
#define atomstruct_chain

#include "imex.h"
#include "polymer.h"
#include "StructureSeq.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX Chain: public StructureSeq {
public:
    Chain(const ChainID& chain_id, Structure* as, PolymerType pt = PT_NONE);
    virtual ~Chain();

    bool  is_chain() const { return !is_sequence() && _is_chain; }
    void  set_chain_id(ChainID chain_id);
};

}  // namespace atomstruct

#endif  // atomstruct_chain
