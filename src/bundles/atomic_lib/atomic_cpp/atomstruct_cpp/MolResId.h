// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_MolResId
#define atomstruct_MolResId

#include <string>
#include <sstream>

#include "imex.h"
#include "Residue.h"
#include "string_types.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX MolResId {
    // convenience class for testing chain/number/insert-code equality
public:
    int    number;
    ChainID chain;
    char insert;
    MolResId() {};
    MolResId(const ChainID& c, int n, char ic) {
        chain = c;
        number = n;
        insert = ic;
    };
    MolResId(char c, int n, char ic) :
        MolResId(std::string(1, c).c_str(), n, ic) {};
    MolResId(const atomstruct::Residue *r) {
        chain = r->chain_id();
        number = r->number();
        insert = r->insertion_code();
    };
    bool operator==(const MolResId &rid) const {
        return rid.number == number && rid.chain == chain && rid.insert == insert;
    }
    bool operator!=(const MolResId &rid) const {
        return rid.number != number || rid.chain != chain || rid.insert != insert;
    }
    bool operator<(const MolResId &rid) const {
        return chain < rid.chain || 
            (chain == rid.chain && (number < rid.number || 
            (number == rid.number && insert < rid.insert)));
    }
};

inline std::ostream & operator<<(std::ostream &os, const MolResId &rid) {
    os << rid.number;
    if (rid.insert != ' ')
        os << rid.insert;
    if (rid.chain != " ")
        os << "." << rid.chain;
    return os;
}

}  // namespace atomstruct

#endif  // atomstruct_MolResId
