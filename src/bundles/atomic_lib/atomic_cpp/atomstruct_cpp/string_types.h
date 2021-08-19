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

#ifndef atomstruct_string_types
#define atomstruct_string_types

#include <string>		// use std::string
#include <chutil/CString.h>

namespace atomstruct {

using chutil::CString;

// len param includes null
#if 0
typedef CString<5, 'A', 't', 'o', 'm', ' ', 'N', 'a', 'm', 'e'>  AtomName; // if changed to string,
#else
typedef std::string AtomName;
#endif
// pdb reader's canonicalize_atom_name needs to be changed accordingly
typedef CString<5, 'A', 't', 'o', 'm', ' ', 'T', 'y', 'p', 'e'>  AtomType;
typedef std::string ChainID;
typedef std::string ResName;

}  // namespace atomstruct

#endif  // atomstruct_string_types
