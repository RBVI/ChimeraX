// vi: set expandtab shiftwidth=4 softtabstop=4:

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

// ----------------------------------------------------------------------------
// Define array index numeric types.
//
#ifndef INDEX_TYPES_HEADER_INCLUDED
#define INDEX_TYPES_HEADER_INCLUDED

#include <cstdint>	// use std::int64_t

typedef int VIndex;		// Surface vertex index. Triangle array uses int32 values.
typedef std::int64_t TIndex;	// Triangle index.
  
typedef unsigned int AIndex;	// Index along one axis of 3D array
typedef std::int64_t GIndex;	// For offsets into 3D arrays

#endif
