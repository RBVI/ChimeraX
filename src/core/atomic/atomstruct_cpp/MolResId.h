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

#ifndef atomstruct_MolResId
#define atomstruct_MolResId

#include <string>
#include <sstream>

#include "imex.h"
#include "Residue.h"
#include "string_types.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX MolResId {
    // convenience class for testing chain/position/insert-code equality
public:
    int    pos;
    ChainID chain;
    char insert;
    MolResId() {};
    MolResId(const ChainID& c, int p, char ic) {
        chain = c;
        pos = p;
        insert = ic;
    };
    MolResId(char c, int p, char ic) :
        MolResId(std::string(1, c).c_str(), p, ic) {};
    MolResId(const atomstruct::Residue *r) {
        chain = r->chain_id();
        pos = r->position();
        insert = r->insertion_code();
    };
    bool operator==(const MolResId &rid) const {
        return rid.pos == pos && rid.chain == chain && rid.insert == insert;
    }
    bool operator!=(const MolResId &rid) const {
        return rid.pos != pos || rid.chain != chain || rid.insert != insert;
    }
    bool operator<(const MolResId &rid) const {
        return chain < rid.chain || 
            (chain == rid.chain && (pos < rid.pos || 
            (pos == rid.pos && insert < rid.insert)));
    }
};

inline std::ostream & operator<<(std::ostream &os, const MolResId &rid) {
    os << rid.pos;
    if (rid.insert != ' ')
        os << rid.insert;
    if (rid.chain != " ")
        os << "." << rid.chain;
    return os;
}

}  // namespace atomstruct

#endif  // atomstruct_MolResId
