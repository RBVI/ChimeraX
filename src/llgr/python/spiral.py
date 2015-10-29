# vim: set expandtab shiftwidth=4 softtabstop=4:
#	Copyright 2004-2008 by the Regents of the University of California.
#	All rights reserved.  This software provided pursuant to a
#	license agreement containing restrictions on its disclosure,
#	duplication and use.  This notice must be embedded in or
#	attached to all copies, including partial copies, of the
#	software or any revisions or derivations thereof.
#
#	$Id: spiral.cpp 34118 2011-07-27 22:42:50Z gregc $
#
#
#   The spiral sphere points algorithm is from:
#
#	E. B. Saff and A. B. J. Kuijlaars, "Distributing Many Points on a
#	Sphere," The Mathematical Intelligencer, Vol. 19 (1997) No. 1, pp.
#	5-11.  It's on the web at
#	<http://www.math.vanderbilt.edu/~esaff/distmany.pdf>.
#
#   Algorithm from the paper:
#
#	"One cuts the global with N horizontal planes spaced 2 / (N - 1)
#	units apart, forming N circles of latitude on the sphere [the first
#	and last of these are degenerate circles (points) consisting of the
#	south and north poles].  Each latitude contains precisely one spiral
#	point.  To obtain the kth spiral point, one proceeds upward from the
#	(k - 1)st point (theta sub k - 1, phi sub k - 1) along a great
#	circle (meridian) to the next latitude and travels counterclockwise
#	along it for a fixed distance (independent of k) to arrive at the
#	kth point (theta sub k, phi sub k)."
#
#    The triangle tesselation algorithm was written by Greg Couch, UCSF
#    Computer Graphics Lab, gregc@cgl.ucsf.edu, January 2004.
#
#    The source code was inspired by Joesph O'Rourke and Min Xu's code
#    for the textbook: "Computational Geometry in C", 1997.  Within that
#    code is an implementation of the recurrence relation given in Saff
#    and Kuijlaars.
#
#    Knud Thomsen modification is from:
#    http://groups.google.com/group/sci.math/browse_thread/thread/983105fb1ced42c/e803d9e3e9ba3d23#e803d9e3e9ba3d23
#
#    Anton Sherwood golden ration modification is available from:
#    http://www.bendwavy.org/pack/pack.htm
#    http://www.cgafaq.info/wiki/Evenly_Distributed_Points_On_Sphere
#

#undef TEST_MAX_SPREAD
#undef TEST_TRIANGLE_SIZE
#undef KNUD_THOMSEN

import math, numpy

MIN_VERTICES	= 8
MAX_VERTICES	= 65536    # because index is unsigned short
MAX_TRIANGLES	= 2 * MAX_VERTICES - 4
MAX_VERTEX_SPREAD = 447	# in a triangle along equator

def points(N):
	if N < MIN_VERTICES:
		N = MIN_VERTICES
	elif N >= MAX_VERTICES:
		N = MAX_VERTICES - 1

	pts = numpy.zeros((N, 3), dtype=numpy.float32)
	phis = numpy.zeros((N,), dtype=numpy.float32)
	thetas = numpy.zeros((N,), dtype=numpy.float32)

	pts[0][0] = 0
	pts[0][1] = 0
	pts[0][2] = -1
	thetas[0] = math.pi

	step_factor = 3.6 / math.sqrt(N)
	scale_factor = 2.0 / (N - 1)

#ifdef KNUD_THOMSEN
#	a = 1 - 1 / (N - 3)
#	b = .5 * (N + 1) / (N - 3)
#endif

	prev_phi = 0.
	for k in range(1, N - 1):
		# compute theta -- it ranges from pi to zero
#ifdef KNUD_THOMSEN
#		kp = a * k + b
#		h = -1 + kp * scale_factor
#else
		h = -1 + k * scale_factor
#endif
		# sanity check h before calling acos
		if h < -1:
			h = -1
		elif h > 1:
			h = 1
		theta = math.acos(h)
		sinTheta = math.sin(theta)

#if 0
#		# from the paper
#		phi = prev_phi + 3.6 / sqrt(N * (1 - h * h))
#else
		# same as above, noting sqrt(1 - h * h) is sin(theta)
		phi = prev_phi + step_factor / sinTheta
#endif
		if phi > 2 * math.pi:
			phi -= 2 * math.pi
		prev_phi = phi

		pts[k][0] = math.cos(phi) * sinTheta
		pts[k][1] = math.sin(phi) * sinTheta
		pts[k][2] = h	# h is cos(theta)
		phis[k] = phi
		thetas[k] = theta

	pts[N - 1][0] = 0
	pts[N - 1][1] = 0
	pts[N - 1][2] = 1
	phis[N - 1] = 0
	thetas[N - 1] = 0
	return pts, phis, thetas

#
# angle_in_interval returns true if query angle in in (start, stop)
# angle interval.
#

def angle_in_interval(query, start, stop):
	d0 = query - start
	if d0 < 0:
		d0 += 2 * math.pi
	d1 = stop - query
	if d1 < 0:
		d1 += 2 * math.pi
	return d0 <= math.pi and d1 <= math.pi

#
# angle_sdist returns the signed angular distance counterclockwise
# from start to stop
#
def angle_sdist(start, stop):
	d = stop - start
	if d < -math.pi:
		d += 2 * math.pi
	return d

def triangles(phis):
	# lowest numbered vertex is always first
	N = len(phis)
	assert MIN_VERTICES <= N <= MAX_VERTICES
	num_triangles = 2 * N - 4
	t = 0
	if N < 256:
		tris = numpy.zeros((num_triangles, 3), dtype=numpy.uint8)
	elif N < 65536:
		tris = numpy.zeros((num_triangles, 3), dtype=numpy.uint16)
	else:
		tris = numpy.zeros((num_triangles, 3), dtype=numpy.uint32)

	# south pole cap -- triangle fan
	prev_phi = phis[1]
	for k in range(2, N - 2):
		tris[t][0] = 0
		tris[t][1] = k
		tris[t][2] = k - 1
		t += 1
		# Check if next phi interval would bracket the
		# original spoke's phi.  If yes, then terminate
		# the cap.  Add a fudge factor to terminate a
		# spoke early for more uniform triangles.
		#
		if angle_in_interval(prev_phi, phis[k], phis[k + 1] + math.pi / k):
			tris[t][0] = 0
			tris[t][1] = 1
			tris[t][2] = k
			t += 1
			break
	# k is last spiral point used

	# north pole cap -- triangle fan
	# Place these triangles at end of list, so triangles are
	# ordered from south to north pole.
	t2 = num_triangles - 1
	j = N - 2
	prev_phi = phis[j]
	for j in range(j - 1, 1, -1):
		tris[t2][0] = j
		tris[t2][1] = j + 1
		tris[t2][2] = N - 1
		t2 -= 1
		# see comment for south pole cap above
		if angle_in_interval(prev_phi, phis[j - 1] - math.pi / (N - j), phis[j]):
			tris[t2][0] = j
			tris[t2][1] = N - 1
			tris[t2][2] = N - 2
			t2 -= 1
			break
	# j - 1 is end of unused spiral points

	# triangle strip around the middle
	# i and k are nearby longitudinally, and start out as
	# the unconnected longitudinal edge from the south pole cap,
	# and are updated to the next unconnected edge
	i = 1
	tris[t][0] = k
	tris[t][1] = i
	k += 1
	tris[t][2] = k
	t += 1
	while i < j and k < N - 2:
		# figure out next two vertices
		# might need a degenerate triangle
		dist_kk = angle_sdist(phis[k], phis[k + 1])
		dist_ki = angle_sdist(phis[k], phis[i + 1])
		dist_ik = angle_sdist(phis[i], phis[k + 1])
		tris[t][0] = i
		if dist_kk >= dist_ki:
			if dist_ik < dist_ki:
				# v0 = i
				i += 1
				tris[t][1] = i
				tris[t][2] = k
				t += 1

				tris[t][0] = i
				tris[t][1] = k + 1
				tris[t][2] = k
				k += 1
				t += 1
			else:
				# v0 = i
				i += 1
				tris[t][1] = i
				tris[t][2] = k
				t += 1
		else:
			if dist_ki < dist_ik:
				# v0 = i
				i += 1
				tris[t][1] = i
				tris[t][2] = k
				t += 1

				tris[t][0] = i
				tris[t][1] = k + 1
				tris[t][2] = k
				k += 1
				t += 1
			else:
				# v0 = i
				tris[t][1] = k + 1
				tris[t][2] = k
				k += 1
				t += 1

	while i != j or k != N - 2:
		k = N - 2
		tris[t][0] = i
		i += 1
		tris[t][1] = i
		tris[t][2] = k
		t += 1

	assert t == t2 + 1
	return tris

if __name__ == "__main__":
	# output VRML file
	NUM_PTS = 1000
	pts, phis, thetas = points(NUM_PTS)
	tris = triangles(phis)
	print(
"""#VRML V2.0 utf8
Shape {
 appearance Appearance {
  material Material {
   diffuseColor 0.8 0.4 0.2
  }
 }
 geometry IndexedFaceSet {
  coordIndex [""")
	for i in range(len(tris)):
		print("   %d %d %d -1," % (tris[i][0], tris[i][1], tris[i][2]))
	print(
"""  ]
  coord Coordinate {
   point [""")
	for i in range(len(pts)):
		print("    %g %g %g," % (pts[i][0], pts[i][1], pts[i][2]))
	print(
"""   ]
  }
  normal Normal {
   vector [""")
	for i in range(len(pts)):
		print("    %g %g %g," % (pts[i][0], pts[i][1], pts[i][2]))
	print(
"""   ]
  }
 }
}""")

"""

struct nvinfo {
	GLfloat nx, ny, nz
	GLfloat x, y, z
}

int
draw_sphere(N, radius)
	# N -- number of points
	# radius -- radius of sphere
{
	if (N < MIN_VERTICES)
		N = MIN_VERTICES
	else if (N > MAX_VERTICES)
		N = MAX_VERTICES

	pts, phis, _ = points(N)
	tris = triangles(N, phis)
	if not tris:
		return 0
	num_triangles = 2 * N - 4

	# TODO: with vertex buffer objects, get array memory from OpenGL
	nvinfo *array = new nvinfo [N]
	for (unsigned int i = 0; i < N; ++i) {
		array[i].nx = static_cast<GLfloat>(pts[i].x)
		array[i].ny = static_cast<GLfloat>(pts[i].y)
		array[i].nz = static_cast<GLfloat>(pts[i].z)
		array[i].x = static_cast<GLfloat>(radius * pts[i].x)
		array[i].y = static_cast<GLfloat>(radius * pts[i].y)
		array[i].z = static_cast<GLfloat>(radius * pts[i].z)
	}
	glInterleavedArrays(GL_N3F_V3F, 0, array)

	#
	# glGet(GL_MAX_ELEMENTS_VERTICES)
	#	== 4096 on NVidia Quadro4 900
	#	== 2147483647 on ATI Radeon 9500 Pro
	# glGet(GL_MAX_ELEMENTS_INDICES)
	#	== 4096 on NVidia Quadro4 900
	#	== 65535 on ATI Radeon 9500 Pro
	#

	glDrawElements(GL_TRIANGLES, 3 * num_triangles,
						GL_UNSIGNED_SHORT, tris)

	return num_triangles
}
"""
