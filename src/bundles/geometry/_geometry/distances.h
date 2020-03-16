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
// Routines to calculate distances from arrays of points to lines or an origin.
//
#ifndef DISTANCES_HEADER_INCLUDED
#define DISTANCES_HEADER_INCLUDED

namespace Distances
{

void distances_from_origin(float points[][3], int n, float origin[3],
			   float distances[]);
void distances_perpendicular_to_axis(float points[][3], int n,
				     float origin[3], float axis[3],
				     float distances[]);
void distances_parallel_to_axis(float points[][3], int n,
				float origin[3], float axis[3],
				float distances[]);

float maximum_norm(float points[][3], int n, float tf[3][4]);

}  // end of namespace Distances

#endif
