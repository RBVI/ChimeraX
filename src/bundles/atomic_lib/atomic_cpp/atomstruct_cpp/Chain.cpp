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
    _is_chain = true;
}

Chain::~Chain()
{
    DestructionUser(this);
    // demote_to_sequence may have already called this
    if (is_chain())
        _structure->change_tracker()->add_deleted(_structure, this);
}
}  // namespace atomstruct
