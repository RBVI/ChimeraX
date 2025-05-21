// vi: set expandtab shiftwidth=4 softtabstop=4:

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
