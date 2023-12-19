// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
#include <math.h>			// use sqrt()
#include "distances.h"

namespace Distances
{

// ----------------------------------------------------------------------------
//
void distances_from_origin(float points[][3], int n, float origin[3],
			   float distances[])
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float dx = p[0] - x0, dy = p[1] - y0, dz = p[2] - z0;
      distances[k] = static_cast<float>(sqrt(dx*dx + dy*dy + dz*dz));
    }
}

// ----------------------------------------------------------------------------
//
void distances_perpendicular_to_axis(float points[][3], int n,
				     float origin[3], float axis[3],
				     float distances[])
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  float ax = axis[0], ay = axis[1], az = axis[2];
  float norm = static_cast<float>(sqrt(ax*ax + ay*ay + az*az));
  if (norm != 0)
    { ax /= norm; ay /= norm ; ax /= norm; }
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float dx = p[0] - x0, dy = p[1] - y0, dz = p[2] - z0;
      float da = dx*ax + dy*ay + dz*az;
      float d2 = dx*dx + dy*dy + dz*dz - da*da;
      if (d2 < 0) d2 = 0;
      distances[k] = static_cast<float>(sqrt(d2));
    }
}

// ----------------------------------------------------------------------------
//
void distances_parallel_to_axis(float points[][3], int n,
				float origin[3], float axis[3],
				float distances[])
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  float ax = axis[0], ay = axis[1], az = axis[2];
  float norm = static_cast<float>(sqrt(ax*ax + ay*ay + az*az));
  if (norm != 0)
    { ax /= norm; ay /= norm ; ax /= norm; }
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float dx = p[0] - x0, dy = p[1] - y0, dz = p[2] - z0;
      distances[k] = dx*ax + dy*ay + dz*az;
    }
}

// Return max(|Tp|)
float maximum_norm(float points[][3], int n, float t[3][4])
{
  float t00 = t[0][0], t01 = t[0][1], t02 = t[0][2], t03 = t[0][3];
  float t10 = t[1][0], t11 = t[1][1], t12 = t[1][2], t13 = t[1][3];
  float t20 = t[2][0], t21 = t[2][1], t22 = t[2][2], t23 = t[2][3];
  float d2max = 0;
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float x = p[0], y = p[1], z = p[2];
      float tpx = t00*x + t01*y + t02*z + t03;
      float tpy = t10*x + t11*y + t12*z + t13;
      float tpz = t20*x + t21*y + t22*z + t23;
      float d2 = tpx*tpx + tpy*tpy + tpz*tpz;
      if (d2 > d2max)
	d2max = d2;
    }
  float d = sqrt(d2max);
  return d;
}

}  // end of namespace Distances
