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

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Chain.h"
#include "ChangeTracker.h"
#include "destruct.h"
#include "Structure.h"

namespace atomstruct {

Chain::Chain(const ChainID& chain_id, Structure* s, PolymerType pt): StructureSeq(chain_id, s, pt)
{
    _structure->change_tracker()->add_created(_structure, this);
}

Chain::~Chain()
{
    // demote_to_sequence may have already called this
    if (is_chain())
        _structure->change_tracker()->add_deleted(_structure, this);
}
}  // namespace atomstruct
