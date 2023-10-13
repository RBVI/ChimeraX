// vi: set expandtab shiftwidth=4 softtabstop=4:

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
// Routines to calculate distances from arrays of points to lines or an origin.
//
#ifndef FITTING_HEADER_INCLUDED
#define FITTING_HEADER_INCLUDED

#include <cstdint>	// use std::int64_t

namespace Fitting
{
void correlation_gradient(float point_weights[], int64_t n,
			  float values[], float gradients[][3],
			  bool about_mean, float *gradient_ret);
void torque(float points[][3], int64_t n, float *point_weights,
	    float forces[][3], float center[3], float *torque_ret);
void torques(float points[][3], int64_t n, float center[3], float forces[][3],
	     float torques[][3]);
void correlation_torque(float points[][3], int64_t n, float point_weights[],
			float values[], float gradients[][3], float center[3],
			bool about_mean, float *torque_ret);
void correlation_torque2(float point_weights[], int64_t n,
			 float values[], float rxg[][3],
			 bool about_mean, float *torque_ret);
	     

}  // end of namespace Fitting

#endif
