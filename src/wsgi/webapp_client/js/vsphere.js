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

var vsphere;

(function () {
"use strict";

vsphere = function (fxy, txy) {
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
		f = vec3.fromValues(fxy.x, fxy.y, Math.sqrt(1 - d1));
	} else {
		d1 = Math.sqrt(d1);
		f = vec3.fromValues(fxy.x / d1, fxy.y / d1, 0);
	}
	if (d2 < 1) {
		t = vec3.fromValues(txy.x, txy.y, Math.sqrt(1 - d2));
	} else {
		d2 = Math.sqrt(d2);
		t = vec3.fromValues(txy.x / d2, txy.y / d2, 0);
	}

	/*
	 * If the positions normalize to the same place we just punt.
	 * We don't even bother to put in the identity matrix.
	 */
	if (Math.abs(f[0] - t[0]) < 1e-12
	&& Math.abs(f[1] - t[1]) < 1e-12
	&& Math.abs(f[2] - t[2]) < 1e-12)
		throw "no change";

	a = vec3.cross(vec3.create(), f, t);
	vec3.normalize(a, a);
	// Don't need to normalize these since the cross product of
	// normal unit vectors is a unit vector
	g = vec3.cross(vec3.create(), a, f);
	u = vec3.cross(vec3.create(), a, t);

	/*
	 * Now assemble them into the inverse matrix (to go from
	 * the from-vector to xyz-space) and the transform matrix
	 * (to go from xyz-space to to-vector).  The product of
	 * the inverse and transformation matrices is the rotation
	 * matrix.
	 */
	m1 = mat4.create();
	m1[0] = t[0]; m1[4] = a[0]; m1[8] = u[0];
	m1[1] = t[1]; m1[5] = a[1]; m1[9] = u[1];
	m1[2] = t[2]; m1[6] = a[2]; m1[10] = u[2];
	m2 = mat4.create();
	m2[0] = f[0]; m2[4] = f[1]; m2[8] = f[2];
	m2[1] = a[0]; m2[5] = a[1]; m2[9] = a[2];
	m2[2] = g[0]; m2[6] = g[1]; m2[10] = g[2];
	return mat4.multiply(mat4.create(), m1, m2);
};

})();
