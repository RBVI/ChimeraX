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
//
#ifndef REFCOUNT_HEADER_INCLUDED
#define REFCOUNT_HEADER_INCLUDED

#include "imex.h"

class ARRAYS_IMEX Reference_Count
{
public:
  Reference_Count();
  Reference_Count(const Reference_Count &);
  Reference_Count &operator=(const Reference_Count &);
  virtual ~Reference_Count();
  int reference_count() const;
private:
  int *ref_count;
};

#endif
