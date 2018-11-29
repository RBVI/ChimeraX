// vi: set sw=4: 
//
#define PY_SSIZE_T_CLEAN
#include "fill_ring.h"
#include <math.h>
#include <arrays/pythonarray.h>		// use float_2d_array_values()
#include <arrays/rcarray.h>		// use FArray

namespace {

typedef std::vector<int> IndexList;     // for compatibility with python_array
typedef std::vector<float> VertexList;  // for compatibility with python_array

// convenience constants and functions
static const size_t X = 0;
static const size_t Y = 1;
static const size_t Z = 2;
static const size_t D = 3;

// Note that Vector is a typedef of an array, which is not a first-class type
// so modifiable Vector arguments are missing the & that a class instance
// would have.
typedef float Vector[3];
typedef Vector Point;
typedef Vector Normal;

inline void
vzero(Vector v)
{
    v[X] = v[Y] = v[Z] = 0;
}

inline float 
vnorm(const Vector v)
{
    return sqrtf(v[X] * v[X] + v[Y] * v[Y] + v[Z] * v[Z]);
}

inline float
vdot(const Vector u, const Vector v)
{
    return u[X] * v[X] + u[Y] * v[Y] + u[Z] * v[Z];
}

inline void
vcross(Vector r, const Vector u, const Vector v)
{
    r[X] = (u[Y] * v[Z]) - (u[Z] * v[Y]);
    r[Y] = (u[Z] * v[X]) - (u[X] * v[Z]);
    r[Z] = (u[X] * v[Y]) - (u[Y] * v[X]);

}

inline void
vincr(Vector u, const Vector v)
{
    u[X] += v[X];
    u[Y] += v[Y];
    u[Z] += v[Z];
}

inline void
vsubtract(Vector r, const Vector u, const Vector v)
{
    r[X] = u[X] - v[X];
    r[Y] = u[Y] - v[Y];
    r[Z] = u[Z] - v[Z];
}

inline void
vassign(Vector u, const Vector v)
{
    u[X] = v[X];
    u[Y] = v[Y];
    u[Z] = v[Z];
}

inline void
vaverage(Vector u, const Vector* vs, ssize_t n)
{
    vzero(u);
    for (auto i = 0; i != n; ++i) {
	vincr(u, vs[i]);
    }
    u[X] /= n;
    u[Y] /= n;
    u[Z] /= n;
}

inline void
vaverage(Vector u, const Vector v0, const Vector v1)
{
    vzero(u);
    vincr(u, v0);
    vincr(u, v1);
    u[X] /= 2;
    u[Y] /= 2;
    u[Z] /= 2;
}

inline void
push_vector(VertexList *vertices, const Vector v)
{
    vertices->push_back(v[0]);
    vertices->push_back(v[1]);
    vertices->push_back(v[2]);
}

void
add_triangle(const Vector v0, const Vector v1, const Vector v2, VertexList *vertices, VertexList* normals, IndexList *indices)
{
    Vector d0, d1, n;
    vsubtract(d0, v0, v1);
    vsubtract(d1, v2, v1);
    vcross(n, d1, d0);
    float len = vnorm(n);
    if (len < 1e-6) {
	// can't skip degenerate triangle because its coordinates
	// may be needed for crease quads
	len = 1;
	n[X] = n[Y] = 0;
	n[Z] = 1;
    }
    n[X] /= len;
    n[Y] /= len;
    n[Z] /= len;
    push_vector(vertices, v0);
    push_vector(vertices, v1);
    push_vector(vertices, v2);
    push_vector(normals, n);
    push_vector(normals, n);
    push_vector(normals, n);
    auto base = indices->size();
    indices->push_back(base + 0);
    indices->push_back(base + 1);
    indices->push_back(base + 2);
}

inline void
add_quad(int v0, int v1, int v2, int v3, IndexList *indices)
{
    // add quad to existing coordinates
    indices->push_back(v0);
    indices->push_back(v2);
    indices->push_back(v1);
    indices->push_back(v0);
    indices->push_back(v3);
    indices->push_back(v2);
}

class Plane
{
public:
    Plane(Point origin, Normal normal);
    Plane(const Vector* xyz, Py_ssize_t n);
    float distance(const float xyz[3]);
private:
    float plane[4];
};

inline
Plane::Plane(Point origin, Normal normal)
{
    float len = vnorm(origin);
    plane[X] = normal[0] / len;
    plane[Y] = normal[1] / len;
    plane[Z] = normal[2] / len;
    plane[D] = -(plane[X] * origin[X] + plane[Y] * origin[Y] + plane[Z] * origin[Z]);
}

inline float
Plane::distance(const float xyz[3])
{
    return plane[X] * xyz[X] + plane[Y] * xyz[Y] + plane[Z] * xyz[Z] + plane[D];
}

Plane::Plane(const Vector* verts, Py_ssize_t nverts)
{
    //
    // Constructor when given a set of points
    // Implementation of Newell's algorithm
    // See Foley, van Dam, Feiner, and Hughes (pp. 476-477)
    // Implementation copied from Filippo Tampieri from Graphics Gems
    //
    ssize_t i;
    Point refpt;
    Normal normal;
    const float *u, *v;
    float len;

    // compute the polygon normal and a reference point on
    // the plane. Note that the actual reference point is
    // refpt / nverts
    vzero(normal);
    vzero(refpt);
    for(i = 0; i < nverts; i++) {
        u = verts[i];
        v = verts[(i + 1) % nverts];
        normal[X] += (u[Y] - v[Y]) * (u[Z] + v[Z]);
        normal[Y] += (u[Z] - v[Z]) * (u[X] + v[X]);
        normal[Z] += (u[X] - v[X]) * (u[Y] + v[Y]);
        vincr(refpt, u);
    }
    /* normalize the polygon normal to obtain the first
       three coefficients of the plane equation
    */
    len = vnorm(normal);
    plane[X] = normal[X] / len;
    plane[Y] = normal[Y] / len;
    plane[Z] = normal[Z] / len;
    /* compute the last coefficient of the plane equation */
    len *= nverts;
    plane[D] = -vdot(refpt, normal) / len;
}

} // namespace

static void
offset_fill(float offset, VertexList* vertices, VertexList* normals, IndexList* triangles)
{
    // Double number of triangles and move vertices by offset along the normal.
    // If offset is greater than zero, it forms a thick filling.
    // Offsetting can introduce gaps between triangles where there are creases,
    // and the calling code is expected to have added quad to compensate.

    // copy and permute vertices, and copy and invert normals
    // to make opposite facing triangles
    auto size = vertices->size();
    for (auto i = 0u; i < size; ++i) {
	vertices->push_back((*vertices)[i]);
	normals->push_back(-(*normals)[i]);
    }
    // make new triangles, and permute indices to maintain handedness
    auto base_coord = size / 3;
    auto num_indices = triangles->size();
    for (auto i = 0u; i < num_indices; i += 3) {
	triangles->push_back(base_coord + i + 1);
	triangles->push_back(base_coord + i + 0);
	triangles->push_back(base_coord + i + 2);
    }

    // offset vertices
    auto count = vertices->size();
    for (auto i = 0u; i < count; ++i) {
	(*vertices)[i] += offset * (*normals)[i];
    }
}

static void
fill_small_ring(const Vector* pts, Py_ssize_t n, float offset, VertexList* vertices, VertexList* normals, IndexList* triangles)
{
    // normals are per-vertex in ChimeraX so replicate vertex for each triangle it is in

    // 3-, 4-, and 5- membered rings
    bool thick = offset > 0;
    switch (n) {
      case 3: {
	unsigned twice = thick ? 2 : 1;
	triangles->reserve(1 * 3 * twice);	// 1 triangles with 3 indices each
	vertices->reserve(3 * 3 * twice);	// 3 vertices with xyz values
	normals->reserve(3 * 3 * twice);	// 3 normals with xyz values
	add_triangle(pts[0], pts[1], pts[2], vertices, normals, triangles);
	break;
      }
      case 4: {
	if (thick) {
	    // (2 triangles + 1 quad) * 2 sides
	    triangles->reserve(8 * 3);		// 8 triangles with 3 indices each
	    vertices->reserve(12 * 3);		// 12 vertices with xyz values
	    normals->reserve(12 * 3);		// 12 normals with xyz values
	} else {
	    triangles->reserve(2 * 3);		// 2 triangles with 3 indices each
	    vertices->reserve(6 * 3);		// 6 vertices with xyz values
	    normals->reserve(6 * 3);		// 6 normals with xyz values
	}
	Vector pa, pb;
	pa[0] = pts[2][X] - pts[0][X];
	pa[1] = pts[2][Y] - pts[0][Y];
	pa[2] = pts[2][Z] - pts[0][Z];
	pb[0] = pts[3][X] - pts[1][X];
	pb[1] = pts[3][Y] - pts[1][Y];
	pb[2] = pts[3][Z] - pts[1][Z];
	float sqdista = pa[X] * pa[X] + pa[Y] * pa[Y] + pa[Z] * pa[Z];
	float sqdistb = pb[X] * pb[X] + pb[Y] * pb[Y] + pb[Z] * pb[Z];
	// sqdistance(p0, p2) < sqdistance(p1, p3)
	if (sqdista < sqdistb) {
	    add_triangle(pts[0], pts[1], pts[2], vertices, normals, triangles);
	    add_triangle(pts[2], pts[3], pts[0], vertices, normals, triangles);
	    if (thick)
		add_quad(0, 2, 3, 5, triangles);
	} else {
	    add_triangle(pts[0], pts[1], pts[3], vertices, normals, triangles);
	    add_triangle(pts[1], pts[2], pts[3], vertices, normals, triangles);
	    if (thick)
		add_quad(1, 2, 5, 3, triangles);
	}
	break;
      }
      case 5: {
	// How to accent sugar pucker twist form:
	//
	//   Walk around sugar ring (5 atoms) looking for the three adjacent
	//   atom "twist" plane that puts one of the other two atoms above
	//   the plane and one below.  There will only be one such plane.
	//
	//   Find the centroid of the two non-planar (twist) atoms along with
	//   the atom further away on the ring.  (The centroid of all 5 atoms
	//   doesn't highlight the plane well enough.)
	//
	//   Project that centroid on to the plane.
	//
	//   Use the projected centroid as the common vertex of 5 triangles that
	//   fill the ring (i.e., every two adjacent atoms and the projected
	//   point).  Two of the triangles will form the plane and the other
	//   three will accent the twist.

	if (thick) {
	    // TODO: add triangles to fill center hole
	    // (5 triangles + 5 quads) * 2 sides
	    triangles->reserve(30 * 3);	// 30 triangles with 3 indices each
	    vertices->reserve(30 * 3);	// 30 vertices with xyz values
	    normals->reserve(30 * 3);	// 30 normals with xyz values
	} else {
	    triangles->reserve(5 * 3);	// 5 triangles with 3 indices each
	    vertices->reserve(15 * 3);	// 15 vertices with xyz values
	    normals->reserve(15 * 3);	// 15 normals with xyz values
	}

	static const float PLANAR_CUTOFF = 0.1f;
	static const float ENVELOPE_RATIO = 3.0f;

	// find twist plane
	static const int indices[] = {
	    0, 1, 2, 3, 4, 0, 1, 2, 3, 4
	};
	float dist3, dist4;
	size_t i;
	for (i = 0; i != 5; ++i) {
	    Point ppts[3];
	    for (auto j = 0; j != 3; ++j)
		vassign(ppts[j], pts[indices[i + j]]);
	    Plane p(ppts, 3);
	    dist3 = p.distance(pts[indices[i + 3]]);
	    dist4 = p.distance(pts[indices[i + 4]]);
	    if (dist3 == 0 || dist4 == 0
	    || (dist3 < 0 && dist4 > 0)
	    || (dist3 > 0 && dist4 < 0))
		break;
	}
	// figure out center point
	Vector center;
	float abs_dist3 = fabsf(dist3);
	float abs_dist4 = fabsf(dist4);
	if (abs_dist3 < PLANAR_CUTOFF
	&& abs_dist4 < PLANAR_CUTOFF) {
	    // planar, center is centroid of pts
	    vaverage(center, pts, n);
	} else if (abs_dist3 < abs_dist4
	&& abs_dist4 / abs_dist3 >= ENVELOPE_RATIO) {
	    // envelope, center is mid-point of separating edge
	    vaverage(center, pts[indices[i]], pts[indices[i + 3]]);
	} else if (abs_dist4 < abs_dist3
	&& abs_dist3 / abs_dist4 >= ENVELOPE_RATIO) {
	    // envelope, center is mid-point of separating edge
	    vaverage(center, pts[indices[i + 2]], pts[indices[i + 4]]);
	} else { // (dist3 < 0 && dist4 > 4) or (dist3 > 0 && dist4 < 4)
	    // twist, center is placed in twist plane near twist pts
	    Point ppts[3];
	    vassign(ppts[0], pts[indices[i + 1]]);
	    vassign(ppts[1], pts[indices[i + 3]]);
	    vassign(ppts[2], pts[indices[i + 4]]);
	    vaverage(center, ppts, 3);
	}
	// triangles = ((0, 1, 5), (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 0, 5))
	add_triangle(pts[0], pts[1], center, vertices, normals, triangles);
	add_triangle(pts[1], pts[2], center, vertices, normals, triangles);
	add_triangle(pts[2], pts[3], center, vertices, normals, triangles);
	add_triangle(pts[3], pts[4], center, vertices, normals, triangles);
	add_triangle(pts[4], pts[0], center, vertices, normals, triangles);
	if (thick) {
		add_quad(1, 2, 5, 3, triangles);
		add_quad(4, 5, 8, 6, triangles);
		add_quad(7, 8, 11, 9, triangles);
		add_quad(10, 11, 14, 12, triangles);
		add_quad(13, 14, 2, 0, triangles);
	}
	break;
      }
    }

    if (thick)
	offset_fill(offset, vertices, normals, triangles);
}

void
fill_6ring(const Vector* pts, float offset, size_t anchor, VertexList* vertices, VertexList* normals, IndexList* triangles)
{
    /* 6-membered rings
     *
     * Try to pick a triangulation that would show chair/boat confirmations.
     * Start by selecting outside triangle (0), and making the opposite
     * size a triangle too (1).  And triangulate the middle quad (2) and
     * (3).  So each area has it's own normal.
     *
     *    /\
     *   /0 \
     *  +----+
     *  |2  /|
     *  |  / |
     *  | /  |
     *  |/  3|
     *  +----+
     *   \1 /
     *    \/
     *
     * Selecting triangle (0) can be done in many ways.  
     */

    // assemble the 4 triangles in the diagram
    size_t t[4][3];
    t[0][0] = (anchor + 5) % 6;
    t[0][1] = (anchor + 1) % 6;
    t[0][2] = anchor;
    if (anchor < 3)
	anchor += 3;
    else
	anchor -= 3;
    t[1][0] = (anchor + 1) % 6;
    t[1][1] = anchor;
    t[1][2] = (anchor + 5) % 6;

    t[2][0] = t[0][0];
    t[2][1] = t[1][0];
    t[2][2] = t[0][1];

    t[3][0] = t[1][0];
    t[3][1] = t[1][2];
    t[3][2] = t[0][1];

    bool thick = offset > 0;
    if (thick) {
	// (4 triangles + 3 quads) * 2 sides
	triangles->reserve(20 * 3);	// 20 triangles with 3 indices each
	vertices->reserve(24 * 3);	// 24 vertices with xyz values
	normals->reserve(24 * 3);	// 24 normals with xyz values
    } else {
	triangles->reserve(4 * 3);	// 4 triangles with 3 indices each
	vertices->reserve(12 * 3);	// 12 vertices with xyz values
	normals->reserve(12 * 3);	// 12 normals with xyz values
    }
    for (auto i = 0; i < 4; ++i)
	add_triangle(pts[t[i][0]], pts[t[i][1]], pts[t[i][2]], 
		     vertices, normals, triangles);
#if 0
    if (thick) {
	add_quad(t[][], t[][], t[][], t[][], indices);
	add_quad(t[][], t[][], t[][], t[][], indices);
	add_quad(t[][], t[][], t[][], t[][], indices);
    }
#endif

    if (thick)
	offset_fill(offset, vertices, normals, triangles);
}

const char *fill_small_ring_doc = 
  "fill_small_ring(vertices, offset) -> vertices, normals, triangles\n"
  "\n"
  //"Supported API\n"
  "Construct geometry to fill a 3, 4, or 5 member ring.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "vertices : n by 3 float array\n"
  "  ring vertex x,y,z coordinates.\n"
  "offset : float\n"
  "  Symmetric offset of triangles.  Zero if only one set of triangles.\n"
  "\n"
  "Returns\n"
  "-------\n"
  "vertices : n by 3 float array\n"
  "  vertex x,y,z coordinates.\n"
  "normals : n by 3 float array\n"
  "  x,y,z normals for each vertex.\n"
  "triangles : m by 3 int array\n"
  "  vertex indices specifying 3 vertices for each triangle.\n";

PyObject *
fill_small_ring(PyObject *, PyObject *args, PyObject *keywds)
{
    FArray vertices_in;
    double offset;
    const char *kwlist[] = {"vertices", "offset", NULL};
    if (!PyArg_ParseTupleAndKeywords(
	    args, keywds, const_cast<char *>("O&d"), (char **) kwlist,
	    parse_float_n3_array, &vertices_in, &offset))
	return NULL;
    FArray cvertices_in = vertices_in.contiguous_array();

    VertexList vertices, normals;
    IndexList triangles;
    Py_BEGIN_ALLOW_THREADS
    fill_small_ring(reinterpret_cast<const Vector*>(cvertices_in.values()),
	    cvertices_in.size() / 3, offset, &vertices, &normals, &triangles);
    Py_END_ALLOW_THREADS

    return python_tuple(
	c_array_to_python(static_cast<const float*>(&vertices[0]), vertices.size() / 3, 3),
	c_array_to_python(static_cast<const float*>(&normals[0]), normals.size() / 3, 3),
	c_array_to_python(static_cast<const int*>(&triangles[0]), triangles.size() / 3, 3)
    );
}

const char *fill_6ring_doc = 
  "fill_6ring(vertices, offset, anchor_corner) -> vertices, normals, triangles\n"
  "\n"
  //"Supported API\n"
  "Construct geometry to fill a 6 member ring.\n"
  "Implemented in C++.\n"
  "\n"
  "Parameters\n"
  "----------\n"
  "vertices : n by 3 float array\n"
  "  ring vertex x,y,z coordinates.\n"
  "offset : float\n"
  "  Symmetric offset of triangles.  Zero if only one set of triangles.\n"
  "anchor_corner : int\n"
  "\n"
  "Returns\n"
  "-------\n"
  "vertices : n by 3 float array\n"
  "  vertex x,y,z coordinates.\n"
  "normals : n by 3 float array\n"
  "  x,y,z normals for each vertex.\n"
  "triangles : m by 3 int array\n"
  "  vertex indices specifying 3 vertices for each triangle.\n";

PyObject *
fill_6ring(PyObject *, PyObject *args, PyObject *keywds)
{
    FArray vertices_in;
    double offset;
    int anchor_corner;
    const char *kwlist[] = {"vertices", "offset", "anchor_corner", NULL};
    if (!PyArg_ParseTupleAndKeywords(
	    args, keywds, const_cast<char *>("O&di"), (char **) kwlist,
	    parse_float_n3_array, &vertices_in, &offset, &anchor_corner))
	return NULL;
    FArray cvertices_in = vertices_in.contiguous_array();

    if (cvertices_in.size() != 6 * 3) {
	PyErr_Format(PyExc_ValueError, "Expecting 6 xyz values (18 floats), got %zu",
		     static_cast<size_t>(cvertices_in.size()));
	return NULL;
    }

    VertexList vertices, normals;
    IndexList triangles;
    Py_BEGIN_ALLOW_THREADS
    fill_6ring(reinterpret_cast<const Vector*>(cvertices_in.values()),
	    offset, anchor_corner, &vertices, &normals, &triangles);
    Py_END_ALLOW_THREADS

    return python_tuple(
	c_array_to_python(static_cast<const float*>(&vertices[0]), vertices.size() / 3, 3),
	c_array_to_python(static_cast<const float*>(&normals[0]), normals.size() / 3, 3),
	c_array_to_python(static_cast<const int*>(&triangles[0]), triangles.size() / 3, 3)
    );}
