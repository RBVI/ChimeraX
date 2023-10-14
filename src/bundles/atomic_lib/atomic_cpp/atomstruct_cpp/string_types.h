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
