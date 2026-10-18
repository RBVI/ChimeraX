// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2026 Regents of the University of California. All rights reserved.
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
// Internal to the arrays library; not installed.
//
// Numeric_Array::as_type() casts between every pair of value types, which
// instantiates Array<T>::set<S>() for all 121 (T, S) pairs.  Those loop nests
// are slow to optimize, so instead of instantiating them all in rcarray.cpp,
// each cast_array<T> is explicitly instantiated in its own rcarray_cast_*.cpp
// file so the compiles run in parallel.
//
#ifndef RCARRAY_CAST_HEADER_INCLUDED
#define RCARRAY_CAST_HEADER_INCLUDED

#include "rcarray.h"        // use Numeric_Array

namespace Reference_Counted_Array
{

template <class T>
void cast_array(Numeric_Array &na, Numeric_Array::Value_Type type);

#ifdef RCARRAY_CAST_DEFINITION
template <class T>
void cast_array(Numeric_Array &na, Numeric_Array::Value_Type type)
{
  Array<T> a(na.dimension(), na.sizes());
  call_template_function(a.template set, na.value_type(), (na));
  Numeric_Array nat(type, a);
  na = nat;
}

#define RCARRAY_INSTANTIATE_CAST(T) \
  template void cast_array<T>(Numeric_Array &na, Numeric_Array::Value_Type type);
#endif

}  // end of namespace Reference_Counted_Array

#endif
