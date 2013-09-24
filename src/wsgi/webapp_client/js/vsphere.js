/*
 * Copyright (c) 1989-2011 The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the University of California, San Francisco.  The name of the
 * University may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 * $Id: vsphere.c,v 1.3 89/11/09 15:45:26 conrad Exp $
 */
/*
 * vsphere:
 *	Convert X-Y coordinates into a 4x4 rotation matrix
 */
function vsphere(fxy, txy)
{
	var	i;
	var	d1, d2;
	var	f, t, a, g, u;
	var	m1, m2;

	/*
	 * First construct the unit vectors and computing
	 * the normal unit vectors.  If we cross from outside
	 * the sphere to inside the sphere (or vice versa), then
	 * we ignore the transition because the discontinuity
	 * tends to make molecules jump.
	 */
	d1 = fxy.x * fxy.x + fxy.y * fxy.y;
	d2 = txy.x * txy.x + txy.y * txy.y;
	if (d1 > 1 && d2 < 1)
		throw "outside to inside";
	if (d2 > 1 && d1 < 1)
		throw "inside to outside";
	if (d1 < 1) {
		f = new TSM.vec3(fxy.x, fxy.y, Math.sqrt(1 - d1));
	} else {
		d1 = Math.sqrt(d1);
		f = new TSM.vec3(fxy.x / d1, fxy.y / d1, 0);
	}
	if (d2 < 1) {
		t = new TSM.vec3(txy.x, txy.y, Math.sqrt(1 - d2));
	} else {
		d2 = Math.sqrt(d2);
		t = new TSM.vec3(txy.x / d2, txy.y / d2, 0);
	}

	/*
	 * If the positions normalize to the same place we just punt.
	 * We don't even bother to put in the identity matrix.
	 */
	if (f[0] == t[0] && f[1] == t[1] && f[2] == t[2])
		throw "no change";

	a = TSM.vec3.cross(f, t);
	a.normalize();
	var m;
	g = TSM.vec3.cross(a, f); /* Don't need to normalize these since the */
	u = TSM.vec3.cross(a, t); /* cross product of normal unit vectors is */
				/* a unit vector */

	/*
	 * Now assemble them into the inverse matrix (to go from
	 * the from-vector to xyz-space) and the transform matrix
	 * (to go from xyz-space to to-vector).  The product of
	 * the inverse and transformation matrices is the rotation
	 * matrix.
	 */
	m1 = new TSM.mat4(	t[0], a[0], u[0], 0,
				t[1], a[1], u[1], 0,
				t[2], a[2], u[2], 0,
				0, 0, 0, 1);
	m2 = new TSM.mat4(	f[0], f[1], f[2], 0,
				a[0], a[1], a[2], 0,
				g[0], g[1], g[2], 0,
				0, 0, 0, 1);
	m = m1.multiply(m2);
	return m;
}
