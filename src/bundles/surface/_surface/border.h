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

// ----------------------------------------------------------------------------
// Compute the intersection of a surface with a plane.
//
#ifndef BORDER_HEADER_INCLUDED
#define BORDER_HEADER_INCLUDED

#include <utility>			// use std::pair
#include <vector>			// use std::vector

#include <arrays/rcarray.h>		// use FArray, IArray

namespace Cap_Calculation
{
typedef std::vector<float> Vertices;
typedef std::vector<int> Triangles;
typedef std::pair<int,int> Loop;
typedef std::vector<Loop> Loops;

// Finds plane intersection with surface as set of polygonal loops.
void calculate_border(float plane_normal[3], float plane_offset,
		      const FArray &varray, const IArray &tarray, /* Surface */
		      Vertices &border_vertices, Loops &loops);

}	// end of namespace Cap_Calculation

#endif
