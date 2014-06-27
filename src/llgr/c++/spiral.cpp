//	Copyright 2004-2008 by the Regents of the University of California.
//	All rights reserved.  This software provided pursuant to a
//	license agreement containing restrictions on its disclosure,
//	duplication and use.  This notice must be embedded in or
//	attached to all copies, including partial copies, of the
//	software or any revisions or derivations thereof.
//
//	$Id: spiral.cpp 34118 2011-07-27 22:42:50Z gregc $

/*
 *   The spiral sphere points algorithm is from:
 *
 *	E. B. Saff and A. B. J. Kuijlaars, "Distributing Many Points on a
 *	Sphere," The Mathematical Intelligencer, Vol. 19 (1997) No. 1, pp.
 *	5-11.  It's on the web at
 *	<http://www.math.vanderbilt.edu/~esaff/distmany.pdf>.
 *
 *   Algorithm from the paper:
 *
 *	"One cuts the global with N horizontal planes spaced 2 / (N - 1)
 *	units apart, forming N circles of latitude on the sphere [the first
 *	and last of these are degenerate circles (points) consisting of the
 *	south and north poles].  Each latitude contains precisely one spiral
 *	point.  To obtain the kth spiral point, one proceeds upward from the
 *	(k - 1)st point (theta sub k - 1, phi sub k - 1) along a great
 *	circle (meridian) to the next latitude and travels counterclockwise
 *	along it for a fixed distance (independent of k) to arrive at the
 *	kth point (theta sub k, phi sub k)."
 *
 *    The triangle tesselation algorithm was written by Greg Couch, UCSF
 *    Computer Graphics Lab, gregc@cgl.ucsf.edu, January 2004.
 *
 *    The source code was inspired by Joesph O'Rourke and Min Xu's code
 *    for the textbook: "Computational Geometry in C", 1997.  Within that
 *    code is an implementation of the recurrence relation given in Saff
 *    and Kuijlaars.
 *
 *    Knud Thomsen modification is from:
 *    http://groups.google.com/group/sci.math/browse_thread/thread/983105fb1ced42c/e803d9e3e9ba3d23#e803d9e3e9ba3d23
 *
 *    Anton Sherwood golden ration modification is available from:
 *    http://www.bendwavy.org/pack/pack.htm
 *    http://www.cgafaq.info/wiki/Evenly_Distributed_Points_On_Sphere
 */

#include <math.h>
#include <assert.h>
#include <algorithm>
#include "spiral.h"

#ifdef _MSC_VER
# undef max
#endif

#ifndef M_PI
#define	 M_PI	3.14159265358979323846
#endif

#undef TEST_MAX_SPREAD
#undef TEST_TRIANGLE_SIZE
#undef KNUD_THOMSEN

namespace llgr {

namespace spiral {

#if 0
# define PI M_PI
#else
# define PI ((float) M_PI)
# define double float
# define sqrt sqrtf
# define sin sinf
# define cos cosf
# define acos acosf
#endif

void
points(unsigned int N, pt_info *pts, double *phis, double *thetas)
{
	unsigned int	k;
	double		prev_phi;
	double		step_factor, scale_factor;

	assert(N > 0);

	if (N >= MAX_VERTICES)
		N = MAX_VERTICES - 1;

	pts[0].x = 0;
	pts[0].y = 0;
	pts[0].z = -1;
	if (phis)
		phis[0] = 0;
	if (thetas)
		thetas[0] = PI;

	step_factor = ((double) 3.6) / sqrt((double) N);
	scale_factor = ((double) 2) / (N - 1);

#ifdef KNUD_THOMSEN
	double a = 1 - 1 / (N - 3);
	double b = .5 * (N + 1) / (N - 3);
#endif

	prev_phi = 0;
	for (k = 1; k < N - 1; ++k) {
		double h, theta, sinTheta, phi;

		/* compute theta -- it ranges from pi to zero */
#ifdef KNUD_THOMSEN
		double kp = a * k + b;
		h = -1 + kp * scale_factor;
#else
		h = -1 + k * scale_factor;
#endif
		/* sanity check h before calling acos */
		if (h < -1)
			h = -1;
		else if (h > 1)
			h = 1;
		theta = acos(h);
		sinTheta = sin(theta);

#if 0
		/* from the paper */
		phi = prev_phi + 3.6 / sqrt(N * (1 - h * h));
#else
		/* same as above, noting sqrt(1 - h * h) is sin(theta) */
		phi = prev_phi + step_factor / sinTheta;
#endif
		if (phi > 2 * PI)
			phi -= 2 * PI;
		prev_phi = phi;

		pts[k].x = cos(phi) * sinTheta;
		pts[k].y = sin(phi) * sinTheta;
		pts[k].z = h;	/* h is cos(theta) */
		if (phis)
			phis[k] = phi;
		if (thetas)
			thetas[k] = theta;
	}

	pts[N - 1].x = 0;
	pts[N - 1].y = 0;
	pts[N - 1].z = 1;
	if (phis)
		phis[N - 1] = 0;
	if (thetas)
		thetas[N - 1] = 0;
}

/*
 * angle_in_interval returns true if query angle in in (start, stop)
 * angle interval.
 */

static int
angle_in_interval(double query, double start, double stop)
{
	double d0, d1;
	d0 = query - start;
	if (d0 < 0)
		d0 += 2 * PI;
	d1 = stop - query;
	if (d1 < 0)
		d1 += 2 * PI;
	return (d0 <= PI && d1 <= PI);
}

/*
 * angle_sdist returns the signed angular distance counterclockwise
 * from start to stop
 */
static double
angle_sdist(double start, double stop)
{
	double d = stop - start;
	if (d < -PI)
		d += 2 * PI;
	return d;
}

tri_info *
triangles(const unsigned int N, pt_info * /*pts*/, double *phis)
{
	/* lowest numbered vertex is always first */
	unsigned int num_triangles = 2 * N - 4;
	unsigned int t = 0;
	tri_info *tris = new tri_info [num_triangles];
	Index i, j;
	unsigned int k;
	double prev_phi;
	unsigned int t2;

	assert(N >= MIN_VERTICES || N <= MAX_VERTICES);

	/* south pole cap -- triangle fan */
	prev_phi = phis[1];
	for (k = 2; k < N - 2; ++k) {
		tris[t].v0 = 0;
		tris[t].v1 = static_cast<Index>(k);
		tris[t].v2 = static_cast<Index>(k - 1);
		++t;
		/* Check if next phi interval would bracket the
		 * original spoke's phi.  If yes, then terminate
		 * the cap.  Add a fudge factor to terminate a
		 * spoke early for more uniform triangles.
		 */
		if (angle_in_interval(prev_phi, phis[k],
						phis[k + 1] + PI / k)) {
			tris[t].v0 = 0;
			tris[t].v1 = 1;
			tris[t].v2 = static_cast<Index>(k);
			++t;
			break;
		}
	}
	/* k is last spiral point used */

	/* north pole cap -- triangle fan */
	/* Place these triangles at end of list, so triangles are
	 * ordered from south to north pole. */
	t2 = num_triangles - 1;
	j = static_cast<Index>(N - 2);
	prev_phi = phis[j];
	for (j = static_cast<Index>(j - 1); j >= 2; --j) {
		tris[t2].v0 = j;
		tris[t2].v1 = static_cast<Index>(j + 1);
		tris[t2].v2 = static_cast<Index>(N - 1);
		--t2;
		/* see comment for south pole cap above */
		if (angle_in_interval(prev_phi, phis[j - 1] - PI / (N - j),
								phis[j])) {
			tris[t2].v0 = j;
			tris[t2].v1 = static_cast<Index>(N - 1);
			tris[t2].v2 = static_cast<Index>(N - 2);
			--t2;
			break;
		}
	}
	/* j - 1 is end of unused spiral points */

	/* triangle strip around the middle */
	/* i and k are nearby longitudinally, and start out as
	 * the unconnected longitudinal edge from the south pole cap,
	 * and are updated to the next unconnected edge */
	i = 1;
	tris[t].v0 = static_cast<Index>(k);
	tris[t].v1 = i;
	tris[t].v2 = static_cast<Index>(++k);
	++t;
	while (i < j && k < N - 2) {
		/* figure out next two vertices */
		/* might need a degenerate triangle */
		double dist_kk = angle_sdist(phis[k], phis[k + 1]);
		double dist_ki = angle_sdist(phis[k], phis[i + 1]);
		double dist_ik = angle_sdist(phis[i], phis[k + 1]);
		tris[t].v0 = i;
		if (dist_kk >= dist_ki) {
			if (dist_ik < dist_ki) {
				/* v0 = i */
				tris[t].v1 = ++i;
				tris[t].v2 = static_cast<Index>(k);
				++t;

				tris[t].v0 = i;
				tris[t].v1 = static_cast<Index>(++k);
				tris[t].v2 = static_cast<Index>(k - 1);
				++t;
			} else {
				/* v0 = i */
				tris[t].v1 = ++i;
				tris[t].v2 = static_cast<Index>(k);
				++t;
			}
		} else {
			if (dist_ki < dist_ik) {
				/* v0 = i */
				tris[t].v1 = ++i;
				tris[t].v2 = static_cast<Index>(k);
				++t;

				tris[t].v0 = i;
				tris[t].v1 = static_cast<Index>(++k);
				tris[t].v2 = static_cast<Index>(k - 1);
				++t;
			} else {
				/* v0 = i */
				tris[t].v1 = static_cast<Index>(++k);
				tris[t].v2 = static_cast<Index>(k - 1);
				++t;
			}

		}
	}
	while (i != j || k != N - 2) {
		k = static_cast<Index>(N - 2);
		tris[t].v0 = i;
		tris[t].v1 = ++i;
		tris[t].v2 = static_cast<Index>(k);
		++t;
	}

	assert(t == t2 + 1);
	return tris;
}

#ifdef OPENGL

namespace {

struct nvinfo {
	GLfloat nx, ny, nz;
	GLfloat x, y, z;
};

} // namespace

int
draw_sphere(unsigned int N, const double r)
	/* N -- number of points */
	/* r -- radius of sphere */
{
	static GLuint	max_vertices, max_indices;

	if (N < MIN_VERTICES)
		N = MIN_VERTICES;
	else if (N > MAX_VERTICES)
		N = MAX_VERTICES;

	pt_info *pts = new pt_info [N];
	double *phis = new double [N];
	points(N, pts, phis, NULL);
	tri_info *tris = triangles(N, pts, phis);
	if (tris == NULL) {
		delete [] pts;
		delete [] phis;
		return 0;
	}
	unsigned int num_triangles = 2 * N - 4;

	/* TODO: with vertex buffer objects, get array memory from OpenGL */
	nvinfo *array = new nvinfo [N];
	for (unsigned int i = 0; i < N; ++i) {
		array[i].nx = static_cast<GLfloat>(pts[i].x);
		array[i].ny = static_cast<GLfloat>(pts[i].y);
		array[i].nz = static_cast<GLfloat>(pts[i].z);
		array[i].x = static_cast<GLfloat>(r * pts[i].x);
		array[i].y = static_cast<GLfloat>(r * pts[i].y);
		array[i].z = static_cast<GLfloat>(r * pts[i].z);
	}
	glInterleavedArrays(GL_N3F_V3F, 0, array);

	/*
	 * glGet(GL_MAX_ELEMENTS_VERTICES)
	 *	== 4096 on NVidia Quadro4 900
	 *	== 2147483647 on ATI Radeon 9500 Pro
	 * glGet(GL_MAX_ELEMENTS_INDICES)
	 *	== 4096 on NVidia Quadro4 900
	 *	== 65535 on ATI Radeon 9500 Pro
	 */

	static bool did_once;
	if (!did_once) {
		did_once = true;
		if (glDrawRangeElements) {
			glGetIntegerv(GL_MAX_ELEMENTS_VERTICES,
				reinterpret_cast<GLint*>(&max_vertices));
			glGetIntegerv(GL_MAX_ELEMENTS_INDICES,
				reinterpret_cast<GLint*>(&max_indices));
#if 0
			printf(" max_vertices: %u\n", max_vertices);
			printf(" max_indices: %u\n", max_indices);
#endif
		}
	}

	if (glDrawRangeElements == NULL) {
		/* fallback to OpenGL 1.1 functionality */
		glDrawElements(GL_TRIANGLES, 3 * num_triangles,
						GL_UNSIGNED_SHORT, tris);
	} else {
		if (max_vertices <= MAX_VERTEX_SPREAD
		|| (N <= max_vertices && 3u * num_triangles <= max_indices)) {
			/* can't reasonably partition, or it all fits */
			glDrawRangeElements(GL_TRIANGLES, 0, N - 1,
				3 * num_triangles, GL_UNSIGNED_SHORT, tris);
		} else {
			/* partition triangles for optimal downloading */
			unsigned int count;
			for (unsigned int i = 0; i < num_triangles; i += count) {
				GLuint start, end;
				count = max_indices / 3;
				if (i + count > num_triangles)
					count = num_triangles - i;
				for (;;) {
					/* Make sure vertex spread fits.
					 * Take into account that v0 is
					 * always less than the other two
					 * triangle vertices and that
					 * consecutive triangles use higher
					 * numbered vertices. */
					start = tris[i].v0;
					end = std::max(tris[i + count - 1].v1,
							tris[i + count - 1].v2);
					if (end - start < max_vertices
					|| /*failsafe*/ count == 1)
						break;
					/* Don't want to spend much time here,
					 * so halve the count (but always at
					 * least one). */
					count = count / 2 + 1;
				}
				glDrawRangeElements(GL_TRIANGLES, start, end,
					3 * count, GL_UNSIGNED_SHORT, tris + i);
			}
		}
	}
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	delete [] tris;
	delete [] array;
	delete [] phis;
	delete [] pts;

	return num_triangles;
}

#endif /* OPENGL */

} // namespace spiral

} // namescape llgr

#ifdef TEST_MAX_SPREAD
int
main()
{
	int i, max_spread;
	const int N = 65536;	/* gives 131068 triangles */
	const int num_triangles = 2 * N - 4;
	pt_info *pts = malloc(N * sizeof (pt_info));
	double *phis = malloc(N * sizeof (double));
	tri_info *tris;

	llgr::spiral::points(N, pts, phis, NULL);
	tris = llgr::spiral::triangles(N, pts, phis);

	max_spread = 0;
	for (i = 0; i < num_triangles; ++i) {
		int d = tris[i].v1 - tris[i].v0;
		if (d > max_spread)
			max_spread = d;
		d = tris[i].v2 - tris[i].v0;
		if (d > max_spread)
			max_spread = d;
#if 0
		if (max_spread == 447) {
			printf("%d: (%d %d %d)\n", i,
					tris[i].v0, tris[i].v1, tris[i].v2);
			break;
		}
#endif
	}
	printf("max spread: %d\n", max_spread);
	return 0;
}

#endif /* TEST_MAX_SPREAD */

#ifdef TEST_TRIANGLE_SIZE
#ifdef WIN32
# define OTF_WRAPPY_DLL 1
#endif
#include <otf/Geom3d.h>
#include <stdio.h>
using otf::Geom3d::Point;
using llgr::spiral::pt_info;
using llgr::spiral::tri_info;
int
main()
{
	const int N = 52;
	const int num_triangles = 2 * N - 4;
	pt_info *pts = (pt_info *) malloc(N * sizeof (pt_info));
	double *phis = (double *) malloc(N * sizeof (double));
	tri_info *tris;
	double area = 0;
	double *areas = (double *) malloc(num_triangles * sizeof (double));

	llgr::spiral::points(N, pts, phis, NULL);
	tris = llgr::spiral::triangles(N, pts, phis);

	for (int i = 0; i < num_triangles; ++i) {
		const pt_info &tmp0 = pts[tris[i].v0];
		Point v0(tmp0.x, tmp0.y, tmp0.z);
		const pt_info &tmp1 = pts[tris[i].v1];
		Point v1(tmp1.x, tmp1.y, tmp1.z);
		const pt_info &tmp2 = pts[tris[i].v2];
		Point v2(tmp2.x, tmp2.y, tmp2.z);
		areas[i] = otf::cross(v2 - v0, v1 - v0).length() / 2;
		area += areas[i];
	}
	printf("%d triangles\n", num_triangles);
	printf("%g total area\n", area);
	printf("%g average area\n", area / num_triangles);
	double mean = area / num_triangles;
	double variance = 0;
	for (int i = 0; i < num_triangles; ++i) {
		variance += (areas[i] - mean) * (areas[i] - mean);
	}
	printf("%g area variance\n", variance);
}
#endif /* TEST_TRIANGLE_SIZE */
