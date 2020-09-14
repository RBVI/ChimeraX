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
#define ARRAYS_EXPORT
#include "refcount.h"

// ----------------------------------------------------------------------------
//
static void decrement_reference_count(int *ref_count);

// ----------------------------------------------------------------------------
//
Reference_Count::Reference_Count()
{
  ref_count = new int;
  *ref_count = 1;
}

// ----------------------------------------------------------------------------
//
Reference_Count::~Reference_Count()
{
  decrement_reference_count(ref_count);
  ref_count = (int *) 0;
}

// ----------------------------------------------------------------------------
//
Reference_Count::Reference_Count(const Reference_Count &rc)
{
  decrement_reference_count(ref_count);
  ref_count = rc.ref_count;
  *ref_count += 1;
}

// ----------------------------------------------------------------------------
//
Reference_Count &Reference_Count::operator=(const Reference_Count &rc)
{
  if (rc.ref_count != ref_count)
    {
      decrement_reference_count(ref_count);
      ref_count = rc.ref_count;
      *ref_count += 1;
    }
  return *this;
}

// ----------------------------------------------------------------------------
//
int Reference_Count::reference_count() const
  { return *ref_count; }

// ----------------------------------------------------------------------------
//
static void decrement_reference_count(int *ref_count)
{
  if (*ref_count == 1)
    delete ref_count;
  else
    *ref_count -= 1;
}
