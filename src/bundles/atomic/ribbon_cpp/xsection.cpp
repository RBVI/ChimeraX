// vi: set expandtab ts=4 sw=4:

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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <arrays/pythonarray.h>

#include <iostream>
#include <algorithm>

#include "parse.h"		// Use parse_residues()
#include "xsection.h"

class Mesh
{
public:
  int num_vertices, num_triangles;
  float *vertices, *normals;
  int *triangles;
  Mesh(int nv)
  {
    num_vertices = nv;
    vertices = new float[3*nv];
    normals = new float[3*nv];
    triangles = nullptr;
    num_triangles = 0;
  }
  ~Mesh()
  {
    delete [] vertices;
    delete [] normals;
    delete [] triangles;
  }
  int *allocate_triangles(int nt)
  {
    triangles = new int [3*nt];
    num_triangles = nt;
    return triangles;
  }
};
  
class Geometry
{
public:
  Geometry() : v_start(0), v_end(0), t_start(0), t_end(0), residue_offset(0) {}
  ~Geometry()
  {
    for (auto mi = meshes.begin() ; mi != meshes.end() ; ++mi)
      delete *mi;
  }
  void add_mesh(Mesh *m)
  {
    meshes.push_back(m);
    v_end += m->num_vertices;
    t_end += m->num_triangles;
    // std::cerr << m->num_vertices << " " << m->num_triangles << std::endl;
  }
  void add_range(int residue_index)
  {
    if (t_end == t_start && v_end == v_start)
      return;
    triangle_ranges.push_back(residue_offset + residue_index);
    triangle_ranges.push_back(t_start);
    triangle_ranges.push_back(t_end);
    triangle_ranges.push_back(v_start);
    triangle_ranges.push_back(v_end);
    t_start = t_end;
    v_start = v_end;
  }
  void set_range_offset(int residue_offset)
  {
    this->residue_offset = residue_offset;
  }
  int num_ranges() const
    { return triangle_ranges.size()/5; }
  void ranges(int *r) const
    { memcpy(r, &triangle_ranges[0], triangle_ranges.size()* sizeof(int)); }
  bool empty() const
    { return t_end == 0; }
  int num_vertices() const
    { return v_end; }
  int num_triangles() const
    { return t_end; }
  void arrays(float *va, float *na, int *ta)
  {
    int v_offset = 0;
    for (auto mi = meshes.begin() ; mi != meshes.end() ; ++mi)
      {
	Mesh *m = *mi;
	int nv3 = 3*m->num_vertices, nt3 = 3*m->num_triangles;
	memcpy(va, m->vertices, nv3 * sizeof(float));
	va += nv3;
	memcpy(na, m->normals, nv3 * sizeof(float));
	na += nv3;
	memcpy(ta, m->triangles, nt3 * sizeof(int));
	for (int i = 0 ; i < nt3 ; ++i)
	  ta[i] += v_offset;
	ta += nt3;
	v_offset += m->num_vertices;
      }
  }
  
private:
  int v_start, v_end;         // for tracking vertex range for each residue
  int t_start, t_end;         // for tracking triangle range for each residue
  std::vector<Mesh *> meshes;   // vertices, normals, and triangles
  std::vector<int> triangle_ranges;  // Ranges for each residue (res_index, ts, te, vs, ve)
  int residue_offset;		// Base residue index when adding a range.
};

class RibbonXSection {
private:
    void _generate_normals();
    Mesh* _extrude_smooth(const float* centers, const float* tangents,
			  const float* normals, int num_pts,
			  bool cap_front, bool cap_back, int offset) const;
    Mesh* _extrude_faceted(const float* centers, const float* tangents,
			   const float* normals, int num_pts,
			   bool cap_front, bool cap_back, int offset) const;
    void _normalize_normals(FArray& v) const;
    void _tessellate();

    FArray xs_coords;
    FArray xs_normals;
    FArray xs_coords2;
    FArray xs_normals2;
    IArray tessellation;
    bool is_arrow;
    bool is_faceted;

public:
    RibbonXSection(FArray* coords,
                   FArray* coords2 = NULL,
                   FArray* normals = NULL,
                   FArray* normals2 = NULL,
                   bool faceted = false,
                   const IArray* tess = NULL);
    virtual ~RibbonXSection();

    Mesh* extrude(const float* centers, const float* tangents,
		  const float* normals, int num_pts,
		  bool cap_front, bool cap_back, int vertex_offset = 0) const;
    RibbonXSection* scale(float x_scale, float y_scale) const;
    RibbonXSection* arrow(float x1_scale, float y1_scale, float x2_scale, float y2_scale) const;
};

typedef std::vector<RibbonXSection *> RibbonXSections;

/*
static void
dump_farray(const char *label, const FArray& v, int nc)
{
    std::cerr << label << "(" << v.size(0) << ", " << nc << "):\n";
    int64_t index[2];
    for (int i = 0; i != v.size(0); ++i) {
        index[0] = i;
        std::cerr << '\t';
        for (int j = 0; j != nc; ++j) {
            index[1] = j;
            std::cerr << ' ' << v.value(index);
        }
        std::cerr << '\n';
    }
}
*/

RibbonXSection::RibbonXSection(FArray* coords, FArray* coords2,
                               FArray* normals, FArray* normals2,
                               bool faceted, const IArray* tess)
{
    // We support two kinds of cross sections: plain and arrow
    // For plain cross sections, "coords" is the 2d coordinates for the entire extrusion
    // For arrow cross sections, "coords" is the 2d coordinates for the front and
    // "coords2" is the coordinates for the back end.  Note that the arrow is actually
    // only covering the front half of the extrusion, with the back half being the
    // same as the back end.  This is done because strands tend to twist at the end
    // and if the arrow narrows across the entire extrusion, the entire arrow looks
    // twisted.  By making it narrow only in the front half, the back half looks like
    // it is turning rather than twisting.
    // We also support two kinds of rendering, faceted and unfaceted.
    // For unfaceted rendering, the same normal, "normals", is used whenever a vertex
    // is used as part of the extrusion.
    // For faceted rendering, the normal for a vertex depends on whether it is the
    // lead or trail vertex of a segment.  If a vertex is the lead vertex, "normals"
    // is used for its normal; if it is the trail vertex, "normals2" is used.
    // Note that although xs_coords2 and xs_normals2 both have "2" in their names,
    // they are used completely unrelatedly.

    import_array1();
    if (coords != NULL && coords->dimension() == 0)
        coords = NULL;
    if (coords2 != NULL && coords2->dimension() == 0)
        coords2 = NULL;
    if (normals != NULL && normals->dimension() == 0)
        normals = NULL;
    if (normals2 != NULL && normals2->dimension() == 0)
        normals2 = NULL;
    if (coords == NULL)
        throw std::invalid_argument("no ribbon cross section coordinates");
    if (coords->dimension() != 2 || coords->size(1) != 2)
        throw std::invalid_argument("bad ribbon cross section dimensions");
// std::cerr << "RibbonXSection\n";
    is_arrow = false;
    xs_coords = *coords;
    if (normals) {
        xs_normals = *normals;
    }
    if (coords2) {
        xs_coords2 = *coords2;
        is_arrow = true;
    }
    if (normals2) {
        xs_normals2 = *normals2;
    }
    if (normals == NULL) {
        is_faceted = faceted;
        _generate_normals();
    }
    else if (normals2 == NULL) {
        is_faceted = false;
        _normalize_normals(xs_normals);
    }
    else {
        is_faceted = true;
        _normalize_normals(xs_normals);
        _normalize_normals(xs_normals2);
    }
// std::cerr << "is_faceted: " << is_faceted << '\n';
// dump_farray("xs_coords", xs_coords, 2);
// dump_farray("xs_normals", xs_normals, 2);
// if (xs_coords2.dimension() != 0)
// dump_farray("xs_coords2", xs_coords2, 2);
// if (xs_normals2.dimension() != 0)
// dump_farray("xs_normals2", xs_normals2, 2);
    if (tess && tess->dimension() != 0)
        tessellation = *tess;
    else
        _tessellate();
}

RibbonXSection::~RibbonXSection()
{
    return;
}

Mesh*
RibbonXSection::extrude(const float* centers, const float* tangents,
                        const float* normals, int num_pts,
                        bool cap_front, bool cap_back, int vertex_offset) const
{
    if (is_faceted)
        return _extrude_faceted(centers, tangents, normals, num_pts,
                                cap_front, cap_back, vertex_offset);
    else
        return _extrude_smooth(centers, tangents, normals, num_pts,
                               cap_front, cap_back, vertex_offset);
}

RibbonXSection*
RibbonXSection::scale(float x_scale, float y_scale) const
{
    FArray coords = FArray(xs_coords.dimension(), xs_coords.sizes());
    FArray normals = FArray(xs_normals.dimension(), xs_normals.sizes());
    FArray normals2;
    if (xs_normals2.dimension() != 0)
        normals2 = FArray(xs_normals2.dimension(), xs_normals.sizes());
    float* oc_values = xs_coords.values();
    float* nc_values = coords.values();
    float* on_values = xs_normals.values();
    float* nn_values = normals.values();
    float* on2_values = xs_normals2.values();
    float* nn2_values = normals2.values();
    int num_coords = xs_coords.size(0);
    for (int i = 0; i != num_coords; ++i) {
        float *oc = oc_values + i * 2;
        float *nc = nc_values + i * 2;
        float *on = on_values + i * 2;
        float *nn = nn_values + i * 2;
        nc[0] = oc[0] * x_scale;
        nc[1] = oc[1] * y_scale;
        nn[0] = on[0] * y_scale;
        nn[1] = on[1] * x_scale;
        if (is_faceted) {
            float *on2 = on2_values + i * 2;
            float *nn2 = nn2_values + i * 2;
            nn2[0] = on2[0] * y_scale;
            nn2[1] = on2[1] * x_scale;
        }
    }
    return new RibbonXSection(&coords, NULL, &normals, &normals2, is_faceted, &tessellation);
}

RibbonXSection*
RibbonXSection::arrow(float x1_scale, float y1_scale, float x2_scale, float y2_scale) const
{
    FArray coords = FArray(xs_coords.dimension(), xs_coords.sizes());
    FArray coords2 = FArray(xs_coords.dimension(), xs_coords.sizes());
    FArray normals = FArray(xs_normals.dimension(), xs_normals.sizes());
    FArray normals2;
    if (xs_normals2.dimension() != 0)
        normals2 = FArray(xs_normals2.dimension(), xs_normals.sizes());
    float* oc_values = xs_coords.values();
    float* nc_values = coords.values();
    float* nc2_values = coords2.values();
    float* on_values = xs_normals.values();
    float* nn_values = normals.values();
    float* on2_values = xs_normals2.values();
    float* nn2_values = normals2.values();
    int num_coords = xs_coords.size(0);
// std::cerr << "is_faceted: " << is_faceted << '\n';
// std::cerr << "oc_values: " << oc_values << " size " << xs_coords.dimension() << '\n';
// std::cerr << "nc_values: " << nc_values << " size " << coords.dimension() << '\n';
// std::cerr << "nc2_values: " << nc2_values << " size " << coords2.dimension() << '\n';
// std::cerr << "on_values: " << on_values << " size " << xs_normals.dimension() << '\n';
// std::cerr << "nn_values: " << nn_values << " size " << normals.dimension() << '\n';
// std::cerr << "on2_values: " << on2_values << " size " << xs_normals2.dimension() << '\n';
// std::cerr << "nn2_values: " << nn2_values << " size " << normals2.dimension() << '\n';
    for (int i = 0; i != num_coords; ++i) {
// std::cerr << "i: " << i << '\n';
        float *oc = oc_values + i * 2;
        float *nc = nc_values + i * 2;
        float *nc2 = nc2_values + i * 2;
        float *on = on_values + i * 2;
        float *nn = nn_values + i * 2;
        nc[0] = oc[0] * x1_scale;
        nc[1] = oc[1] * y1_scale;
// std::cerr << "nc: " << nc[0] << ' ' << nc[1] << '\n';
        nc2[0] = oc[0] * x2_scale;
        nc2[1] = oc[1] * y2_scale;
// std::cerr << "nc2: " << nc2[0] << ' ' << nc2[1] << '\n';
        nn[0] = on[0] * y1_scale;
        nn[1] = on[1] * x1_scale;
// std::cerr << "nn: " << nn[0] << ' ' << nn[1] << '\n';
        if (is_faceted) {
            float *on2 = on2_values + i * 2;
            float *nn2 = nn2_values + i * 2;
            nn2[0] = on2[0] * y1_scale;
            nn2[1] = on2[1] * x1_scale;
// std::cerr << "on: " << on[0] << ' ' << on[1] << '\n';
        }
    }
// std::cerr << "done!\n";
    return new RibbonXSection(&coords, &coords2, &normals, &normals2, is_faceted, &tessellation);
}

static bool
is_concave(float ix, float iy, float jx, float jy, float kx, float ky)
{
    float ijx = ix - jx;
    float ijy = iy - jy;
    float kjx = kx - jx;
    float kjy = ky - jy;
    return (ijx * kjy - ijy * kjx) >= 0;
}

void
RibbonXSection::_generate_normals()
{
    FArray &c = is_arrow ? xs_coords2 : xs_coords;
    int num_coords = c.size(0);
    int64_t index0[2] = { 0, 0 };
    int64_t index1[2] = { 0, 1 };
    if (!is_faceted) {
        xs_normals = FArray(c.dimension(), c.sizes());
        for (int i = 0; i != num_coords; ++i) {
            index0[0] = index1[0] = i;
            float ix = c.value(index0);
            float iy = c.value(index1);
            int j = (i + 1) % num_coords;
            index0[0] = index1[0] = j;
            float jx = c.value(index0);
            float jy = c.value(index1);
            int k = (i + 2) % num_coords;
            index0[0] = index1[0] = k;
            float kx = c.value(index0);
            float ky = c.value(index1);
            index0[0] = index1[0] = j;
            if (is_concave(ix, iy, jx, jy, kx, ky)) {
                xs_normals.set(index0, iy - ky);
                xs_normals.set(index1, kx - ix);
            }
            else {
                xs_normals.set(index0, ky - iy);
                xs_normals.set(index1, ix - kx);
            }
        }
        _normalize_normals(xs_normals);
    }
    else {
        xs_normals = FArray(c.dimension(), c.sizes());
        xs_normals2 = FArray(c.dimension(), c.sizes());
        for (int i = 0; i != num_coords; ++i) {
            index0[0] = index1[0] = i;
            float ix = c.value(index0);
            float iy = c.value(index1);
            int j = (i + 1) % num_coords;
            index0[0] = index1[0] = j;
            float jx = c.value(index0);
            float jy = c.value(index1);
            float dx = jx - ix;
            float dy = jy - iy;
            xs_normals2.set(index0, dy);
            xs_normals2.set(index1, -dx);
            index0[0] = index1[0] = i;
            xs_normals.set(index0, dy);
            xs_normals.set(index1, -dx);
        }
        _normalize_normals(xs_normals);
        _normalize_normals(xs_normals2);
    }
}

Mesh*
RibbonXSection::_extrude_smooth(const float* centers, const float* tangents,
                                const float* normals, int num_pts,
                                bool cap_front, bool cap_back, int offset) const
{
// std::cerr << "extrude_smooth " << xs_coords2.dimension() << "\n";
// dump_farray("centers:", centers, 3);
// dump_farray("tangents:", tangents, 3);
// dump_farray("normals:", normals, 3);
    // Calculate result sizes
    int num_splines = xs_coords.size(0);
    int num_pts_per_spline = num_pts;
    int num_vertices = num_splines * num_pts_per_spline;
    if (cap_front)
        num_vertices += num_splines;
    if (cap_back)
        num_vertices += num_splines;
    // Allocate space for vertices and normals
    Mesh *mesh = new Mesh(num_vertices);
    float *va_data = mesh->vertices;
    float *na_data = mesh->normals;
    int vindex = 0;
    // Compute binormals as cross product of tangents and normals
    float* binormals = new float[3*num_pts];
    for (int i = 0; i != num_pts_per_spline; ++i) {
        float* bv = binormals + i * 3;
        const float* tv = tangents + i * 3;
        const float* nv = normals + i * 3;
        bv[0] =  tv[1]*nv[2]-tv[2]*nv[1];
        bv[1] = -tv[0]*nv[2]+tv[2]*nv[0];
        bv[2] =  tv[0]*nv[1]-tv[1]*nv[0];
    }
    // Compute extruded vertices and normals
    float* xs_coords_values = xs_coords.values();
    float* xs_coords2_values = xs_coords2.values();
    float* xs_normals_values = xs_normals.values();
    for (int j = 0; j != num_splines; ++j) {
        // Figure out if cross section varies over length of extrusion
        // cp = coordinate xs scaling (2 floats scaling normal and binormal)
        // np = normal xs scaling
        // 1 = start, 2 = end, 2 only valid if scale is true
        float *cp1 = xs_coords_values + j * 2;
        float *np = xs_normals_values + j * 2;
        int arrow_length = 0;
        float *cp2 = NULL;
        float c_step[2] = {0,0};
        if (xs_coords2.dimension() != 0) {
            cp2 = xs_coords2_values + j * 2;
            // Arrow only spans half the residue, with latter half
            // being the same as the second set of coords/normals
            // arrow_length = num_pts_per_spline / 2;
            arrow_length = num_pts_per_spline;
            c_step[0] = (cp2[0] - cp1[0]) / arrow_length;
            c_step[1] = (cp2[1] - cp1[1]) / arrow_length;
        }
        // Compute coordinates for xs point j at extrusion step i
        for (int i = 0; i != num_pts_per_spline; ++i) {
            float cn, cb, nn, nb;
            if (arrow_length == 0) {
                cn = cp1[0];
                cb = cp1[1];
            }
            else if (i <= arrow_length) {
                cn = cp1[0] + c_step[0] * i;
                cb = cp1[1] + c_step[1] * i;
            }
            else {
                cn = cp2[0];
                cb = cp2[1];
            }
            nn = np[0];
            nb = np[1];
            const float* center = centers + i * 3;
            const float* normal = normals + i * 3;
            float* binormal = binormals + i * 3;
            float* vap = va_data + vindex * 3;
            float* nap = na_data + vindex * 3;
            for (int k = 0; k != 3; ++k) {
                vap[k] = center[k] + cn * normal[k] + cb * binormal[k];
                // XXX: These normals are not quite right for an arrow because
                // they should be slanted proportionally to the arrow angle.
                // However, to compute them correctly , we would need to compute
                // the path length and width change rates in order to get the
                // correct proportion and the difference visually is not great.
                // So we ignore the problem for now.
                nap[k] = nn * normal[k] + nb * binormal[k];
            }
            vindex += 1;
        }
    }
    delete [] binormals;
    // Allocate space for triangles
    int num_triangles = num_splines * (num_pts_per_spline - 1) * 2;
    if (cap_front)
        num_triangles += tessellation.size(0);
    if (cap_back)
        num_triangles += tessellation.size(0);
    int *ta_data = mesh->allocate_triangles(num_triangles);
    int tindex = 0;
    // Create triangles
    for (int s = 0; s != num_splines; ++s) {
        int i_start = s * num_pts_per_spline + offset;
        int j = (s + 1) % num_splines;
        int j_start = j * num_pts_per_spline + offset;
        for (int k = 0; k != num_pts_per_spline - 1; ++k) {
            ta_data[tindex] = i_start + k + 1;
            ta_data[tindex + 1] = i_start + k;
            ta_data[tindex + 2] = j_start + k;
            tindex += 3;
            ta_data[tindex] = i_start + k + 1;
            ta_data[tindex + 1] = j_start + k;
            ta_data[tindex + 2] = j_start + k + 1;
            tindex += 3;
        }
    }
    offset += num_splines * num_pts_per_spline;
    // Generate caps if necessary
    if (cap_front) {
        const float* tangent = tangents;
        for (int i = 0; i != num_splines; ++i) {
            float *vto = va_data + (vindex + i) * 3;
            float *vfrom = va_data + (i * num_pts_per_spline) * 3;
            vto[0] = vfrom[0];
            vto[1] = vfrom[1];
            vto[2] = vfrom[2];
            float *nto = na_data + (vindex + i) * 3;
            nto[0] = -tangent[0];
            nto[1] = -tangent[1];
            nto[2] = -tangent[2];
        }
        int* tess = tessellation.values();
        for (int i = 0; i != tessellation.size(0); ++i) {
            int* ti = tess + i * 3;
            ta_data[tindex] = ti[2] + offset;
            ta_data[tindex + 1] = ti[1] + offset;
            ta_data[tindex + 2] = ti[0] + offset;
            tindex += 3;
        }
        offset += num_splines;
        vindex += num_splines;
    }
    if (cap_back) {
        const float* tangent = tangents + (num_pts_per_spline - 1) * 3;;
        for (int i = 0; i != num_splines; ++i) {
            float *vto = va_data + (vindex + i) * 3;
            float *vfrom = va_data + (i * num_pts_per_spline +
                                      num_pts_per_spline - 1) * 3;
            vto[0] = vfrom[0];
            vto[1] = vfrom[1];
            vto[2] = vfrom[2];
            float *nto = na_data + (vindex + i) * 3;
            nto[0] = tangent[0];
            nto[1] = tangent[1];
            nto[2] = tangent[2];
        }
        int* tess = tessellation.values();
        for (int i = 0; i != tessellation.size(0); ++i) {
            int* ti = tess + i * 3;
            ta_data[tindex] = ti[0] + offset;
            ta_data[tindex + 1] = ti[1] + offset;
            ta_data[tindex + 2] = ti[2] + offset;
            tindex += 3;
        }
        offset += num_splines;
        vindex += num_splines;
    }
    return mesh;
}

Mesh*
RibbonXSection::_extrude_faceted(const float* centers, const float* tangents,
                                 const float* normals, int num_pts,
                                 bool cap_front, bool cap_back, int offset) const
{
// std::cerr << "extrude_faceted " << xs_coords2.dimension() << "\n";
// dump_farray("xs_coords", xs_coords, 2);
// dump_farray("xs_normals", xs_normals, 2);
// if (xs_coords2.dimension() != 0)
// dump_farray("xs_coords2", xs_coords2, 2);
// if (xs_normals2.dimension() != 0)
// dump_farray("xs_normals2", xs_normals2, 2);
// dump_farray("centers:", centers, 3);
// dump_farray("tangents:", tangents, 3);
// dump_farray("normals:", normals, 3);
    // Calculate result sizes
    int num_splines = xs_coords.size(0);
    int num_pts_per_spline = num_pts;
    int num_vertices = num_splines * num_pts_per_spline * 2;
    if (cap_front)
        num_vertices += num_splines;
    if (cap_back)
        num_vertices += num_splines;
    // Allocate space for vertices and normals
    Mesh *mesh = new Mesh(num_vertices);
    float *va_data = mesh->vertices;
    float *na_data = mesh->normals;
    int vindex = 0;
    // Compute binormals as cross product of tangents and normals
    float* binormals = new float[3*num_pts];
    for (int i = 0; i != num_pts_per_spline; ++i) {
        float* bv = binormals + i * 3;
        const float* tv = tangents + i * 3;
        const float* nv = normals + i * 3;
        bv[0] =  tv[1]*nv[2]-tv[2]*nv[1];
        bv[1] = -tv[0]*nv[2]+tv[2]*nv[0];
        bv[2] =  tv[0]*nv[1]-tv[1]*nv[0];
    }
    // Compute extruded vertices and normals
    float* xs_coords_values = xs_coords.values();
    float* xs_normals_values = xs_normals.values();
    float* xs_coords2_values = xs_coords2.values();
    float* xs_normals2_values = xs_normals2.values();
    for (int j = 0; j != num_splines; ++j) {
        // Figure out if cross section varies over length of extrusion
        // cp = coordinate xs scaling (2 floats scaling normal and binormal)
        // np = normal xs scaling
        // 1 = start, 2 = end, 2 only valid if scale is true
        float *cp1 = xs_coords_values + j * 2;
        float *np1 = xs_normals_values + j * 2;
        float *np2 = xs_normals2_values + j * 2;
        int arrow_length = 0;
        float *cp2 = NULL;
        float c_step[2] = {0,0};
        if (xs_coords2.dimension() != 0) {
            cp2 = xs_coords2_values + j * 2;
            // arrow_length = num_pts_per_spline / 2;
            arrow_length = num_pts_per_spline - 1;
            c_step[0] = (cp2[0] - cp1[0]) / arrow_length;
            c_step[1] = (cp2[1] - cp1[1]) / arrow_length;
        }
        // Compute coordinates for xs point j at extrusion step i
        for (int i = 0; i != num_pts_per_spline; ++i) {
            float cn, cb;
            if (arrow_length == 0) {
                cn = cp1[0];
                cb = cp1[1];
            }
            else if (i <= arrow_length) {
                cn = cp1[0] + c_step[0] * i;
                cb = cp1[1] + c_step[1] * i;
            }
            else {
                cn = cp2[0];
                cb = cp2[1];
            }
            const float* center = centers + i * 3;
            const float* normal = normals + i * 3;
            float* binormal = binormals + i * 3;
            float* vap = va_data + vindex * 3;
            float* nap = na_data + vindex * 3;
            float* vap2 = va_data + (vindex + num_pts_per_spline) * 3;
            float* nap2 = na_data + (vindex + num_pts_per_spline) * 3;
            // Insert the vertex twice into the arrow, once with each normal
            for (int k = 0; k != 3; ++k) {
                vap[k] = vap2[k] = center[k] + cn * normal[k] + cb * binormal[k];
                nap[k] = np1[0] * normal[k] + np1[1] * binormal[k];
                nap2[k] = np2[0] * normal[k] + np2[1] * binormal[k];
            }
            vindex += 1;
        }
        vindex += num_pts_per_spline;   // account for double addition
    }
    delete [] binormals;
    // Allocate space for triangles
    int num_triangles = num_splines * (num_pts_per_spline - 1) * 2;
    if (cap_front)
        num_triangles += tessellation.size(0);
    if (cap_back)
        num_triangles += tessellation.size(0);
    int *ta_data = mesh->allocate_triangles(num_triangles);
    int tindex = 0;
    // Create triangles
    for (int s = 0; s != num_splines; ++s) {
        int i_start = (s * 2) * num_pts_per_spline + offset;
        int j = (s + 1) % num_splines;
        int j_start = (j * 2 + 1) * num_pts_per_spline + offset;
        for (int k = 0; k != num_pts_per_spline - 1; ++k) {
            ta_data[tindex] = i_start + k + 1;
            ta_data[tindex + 1] = i_start + k;
            ta_data[tindex + 2] = j_start + k;
            tindex += 3;
            ta_data[tindex] = i_start + k + 1;
            ta_data[tindex + 1] = j_start + k;
            ta_data[tindex + 2] = j_start + k + 1;
            tindex += 3;
        }
    }
    offset += num_splines * num_pts_per_spline * 2;
    // Generate caps if necessary
    if (cap_front) {
        const float* tangent = tangents;
        for (int i = 0; i != num_splines; ++i) {
            float *vto = va_data + (vindex + i) * 3;
            float *vfrom = va_data + (i * 2 * num_pts_per_spline) * 3;
            vto[0] = vfrom[0];
            vto[1] = vfrom[1];
            vto[2] = vfrom[2];
            float *nto = na_data + (vindex + i) * 3;
            nto[0] = -tangent[0];
            nto[1] = -tangent[1];
            nto[2] = -tangent[2];
        }
        int* tess = tessellation.values();
        for (int i = 0; i != tessellation.size(0); ++i) {
            int* ti = tess + i * 3;
            ta_data[tindex] = ti[2] + offset;
            ta_data[tindex + 1] = ti[1] + offset;
            ta_data[tindex + 2] = ti[0] + offset;
            tindex += 3;
        }
        offset += num_splines;
        vindex += num_splines;
    }
    if (cap_back) {
        const float* tangent = tangents + (num_pts_per_spline - 1) * 3;;
        for (int i = 0; i != num_splines; ++i) {
            float *vto = va_data + (vindex + i) * 3;
            float *vfrom = va_data + (i * 2 * num_pts_per_spline +
                                      num_pts_per_spline - 1) * 3;
            vto[0] = vfrom[0];
            vto[1] = vfrom[1];
            vto[2] = vfrom[2];
            float *nto = na_data + (vindex + i) * 3;
            nto[0] = tangent[0];
            nto[1] = tangent[1];
            nto[2] = tangent[2];
        }
        int* tess = tessellation.values();
        for (int i = 0; i != tessellation.size(0); ++i) {
            int* ti = tess + i * 3;
            ta_data[tindex] = ti[0] + offset;
            ta_data[tindex + 1] = ti[1] + offset;
            ta_data[tindex + 2] = ti[2] + offset;
            tindex += 3;
        }
        offset += num_splines;
        vindex += num_splines;
    }
    return mesh;
}

void
RibbonXSection::_normalize_normals(FArray& v) const
{
    int64_t size = v.size(0);
    int64_t index0[2] = { 0, 0 };
    int64_t index1[2] = { 0, 1 };
    for (int64_t i = 0; i != size; ++i) {
        index0[0] = index1[0] = i;
        float x = v.value(index0);
        float y = v.value(index1);
        float length = sqrt(x * x + y * y);
        if (length > 0) {
            v.set(index0, x / length);
            v.set(index1, y / length);
        }
    }
}

void
RibbonXSection::_tessellate()
{
    // XXX: Assume convex for now
    int64_t num_coords = xs_coords.size(0);
    int64_t size[2] = { num_coords - 2, 3 };
    tessellation = IArray(2, size);
    int* tdata = tessellation.values();
    for (int64_t i = 1; i != num_coords - 1; ++i) {
        tdata[0] = 0;
        tdata[1] = i;
        tdata[2] = i + 1;
        tdata += 3;
    }
}

// -------------------------------------------------------------------------
// ribbon xsection functions

extern "C" int parse_rxsection_pointer(PyObject *arg, RibbonXSection **pointer)
{
  void *p;
  int success = parse_voidp(arg, &p);
  if (success)
    *pointer = static_cast<RibbonXSection *>(p);
  return success;
}

extern "C" int parse_geometry_pointer(PyObject *arg, Geometry **pointer)
{
  void *p;
  int success = parse_voidp(arg, &p);
  if (success)
    *pointer = static_cast<Geometry *>(p);
  else
    {
      PyErr_SetString(PyExc_TypeError,
 		      "parse_geometry_pointer(): Geometry pointer must be an integer");
    }
  return success;
}

extern "C" PyObject *
rxsection_new(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray coords, coords2, normals, normals2;
  IArray tess;
  int faceted = 0;
  const char *kwlist[] = {"coords", "coords2", "normals", "normals2", "faceted", "tess", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("|O&O&O&O&pO&"),
				   (char **)kwlist,
				   parse_float_n2_array, &coords,
				   parse_float_n2_array, &coords2,
				   parse_float_n2_array, &normals,
				   parse_float_n2_array, &normals2,
				   &faceted,
				   parse_int_n3_array, &tess))
    return NULL;

  RibbonXSection *xs = new RibbonXSection(&coords, &coords2, &normals, &normals2, faceted, &tess);
  
  return python_voidp(xs);
}

extern "C" PyObject *
rxsection_delete(PyObject *, PyObject *args, PyObject *keywds)
{
  RibbonXSection *xs;
  const char *kwlist[] = {"xsection", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_rxsection_pointer, &xs))
    return NULL;

  delete xs;
  
  return python_none();
}

extern "C" PyObject *
rxsection_extrude(PyObject *, PyObject *args, PyObject *keywds)
{
  RibbonXSection *xs;
  FArray centers, tangents, normals;
  int cap_front, cap_back;
  Geometry *geom;
  const char *kwlist[] = {"xsection", "centers", "tangents", "normals", "cap_front",
			  "cap_back", "geometry", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&ppO&"),
				   (char **)kwlist,
				   parse_rxsection_pointer, &xs,
				   parse_float_n3_array, &centers,
				   parse_float_n3_array, &tangents,
				   parse_float_n3_array, &normals,
				   &cap_front, &cap_back,
				   parse_geometry_pointer, &geom))
    return NULL;

  if (!centers.is_contiguous() || !tangents.is_contiguous() || !normals.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "rxsection_extrude(): Centers, tangents and normals arrays must be contiguous");
      return NULL;
    }
  
  Mesh *m = xs->extrude(centers.values(), tangents.values(), normals.values(), centers.size(0),
			cap_front, cap_back);
  geom->add_mesh(m);

  return python_none();
}

extern "C" PyObject *
rxsection_scale(PyObject *, PyObject *args, PyObject *keywds)
{
  RibbonXSection *xs;
  float x_scale, y_scale;
  const char *kwlist[] = {"xsection", "x_scale", "y_scale", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ff"),
				   (char **)kwlist,
				   parse_rxsection_pointer, &xs,
				   &x_scale, &y_scale))
    return NULL;

  RibbonXSection *r = xs->scale(x_scale, y_scale);
  return python_voidp(r);
}

extern "C" PyObject *
rxsection_arrow(PyObject *, PyObject *args, PyObject *keywds)
{
  RibbonXSection *xs;
  float x1_scale, y1_scale, x2_scale, y2_scale;
  const char *kwlist[] = {"xsection", "x1_scale", "y1_scale", "x2_scale", "y2_scale", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&ffff"),
				   (char **)kwlist,
				   parse_rxsection_pointer, &xs,
				   &x1_scale, &y1_scale, &x2_scale, &y2_scale))
    return NULL;

  RibbonXSection *r = xs->arrow(x1_scale, y1_scale, x2_scale, y2_scale);
  return python_voidp(r);
}

extern "C" PyObject *
geometry_new(PyObject *, PyObject *args, PyObject *keywds)
{
  const char *kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>(""),
				   (char **)kwlist))
    return NULL;

  Geometry *g = new Geometry();
  return python_voidp(g);
}

extern "C" PyObject *
geometry_delete(PyObject *, PyObject *args, PyObject *keywds)
{
  Geometry *g;
  const char *kwlist[] = {"geometry", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_geometry_pointer, &g))
    return NULL;

  delete g;
  return python_none();
}

extern "C" PyObject *
geometry_add_range(PyObject *, PyObject *args, PyObject *keywds)
{
  Geometry *g;
  int residue_index;
  const char *kwlist[] = {"geometry", "residue_index", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i"),
				   (char **)kwlist,
				   parse_geometry_pointer, &g,
				   &residue_index))
    return NULL;

  g->add_range(residue_index);
  return python_none();
}

extern "C" PyObject *
geometry_set_range_offset(PyObject *, PyObject *args, PyObject *keywds)
{
  Geometry *g;
  int residue_base;
  const char *kwlist[] = {"geometry", "residue_base", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&i"),
				   (char **)kwlist,
				   parse_geometry_pointer, &g,
				   &residue_base))
    return NULL;

  g->set_range_offset(residue_base);
  return python_none();
}

extern "C" PyObject *
geometry_ranges(PyObject *, PyObject *args, PyObject *keywds)
{
  Geometry *g;
  const char *kwlist[] = {"geometry", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_geometry_pointer, &g))
    return NULL;


  int *r;
  PyObject *ranges = python_int_array(g->num_ranges(),5,&r);
  g->ranges(r);
  
  return ranges;
}

extern "C" PyObject *
geometry_empty(PyObject *, PyObject *args, PyObject *keywds)
{
  Geometry *g;
  const char *kwlist[] = {"geometry", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_geometry_pointer, &g))
    return NULL;

  return python_bool(g->empty());
}

extern "C" PyObject *
geometry_arrays(PyObject *, PyObject *args, PyObject *keywds)
{
  Geometry *g;
  const char *kwlist[] = {"geometry", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&"),
				   (char **)kwlist,
				   parse_geometry_pointer, &g))
    return NULL;

  int nv = g->num_vertices(), nt = g->num_triangles();
  float *va, *na;
  PyObject *vertices = python_float_array(nv, 3, &va);
  PyObject *normals = python_float_array(nv, 3, &na);
  int *ta;
  PyObject *triangles = python_int_array(nt, 3, &ta);
  g->arrays(va, na, ta);
  
  return python_tuple(vertices, normals, triangles);
}

// Compute triangle geometry for ribbon.
// Only certain ranges of residues are considered, since not all
// residues need be displayed and also tube helix geometry is created by other code.
// TODO: This routine is taking half the ribbon compute time.  Probably a
//  big contributor is that 17 numpy arrays are being made per residue.
//  Might want to put TriangleAccumulator into C++ to get rid of half of those
//  and have extrude() put results directly into it.
//  Maybe Ribbon spline coords, tangents, normals could use recycled numpy arrays.
static void ribbon_extrusions(const float *coords, const float *tangents, const float *normals,
			      int num_coords, const int *ranges, int num_ranges, int num_res,
			      const RibbonXSections &xs_front, const RibbonXSections &xs_back,
			      Geometry &geometry)
{
  int nsp = num_coords / num_res;  	// Path points per residue
  int nlp = nsp/2, nrp = (nsp + 1)/2;	// Left and right half points per residue
  
  // Each residue has left and right half (also called front and back)
  // with the residue centered in the middle.
  // The two halfs can have different crosssections, e.g. turn and helix.
  // At the ends of the polymer the spline is extended to make the first residue
  // have a left half and the last residue have a right half.
  // If an interior range is shown only half segments are shown at the ends
  // since other code (e.g. tube cylinders) will render the other halfs.
  for (int r = 0 ; r < num_ranges ; ++r)
    {
      int r0 = ranges[2*r], r1 = ranges[2*r+1];

      bool capped = true;
        
      for (int i = r0 ; i <= r1 ; ++i)
	{
	  // Left half
	  bool mid_cap = (xs_front[i] != xs_back[i]);
	  int s = i * nsp;
	  int e = s + nlp + 1;
	  int num_pts = e-s;
	  const float *front_c = coords + 3*s, *front_t = tangents + 3*s, *front_n = normals + 3*s;
	  Mesh *mleft = xs_front[i]->extrude(front_c, front_t, front_n, num_pts, capped, mid_cap);
	  geometry.add_mesh(mleft);
	  
	  // Right half
	  bool next_cap = (i == r1 ? true : (xs_back[i] != xs_front[i + 1]));
	  s = i * nsp + nlp;
	  e = (i < num_res-1 ? s + nrp + 1 : s + nrp);
	  num_pts = e-s;
	  const float *back_c = coords + 3*s, *back_t = tangents + 3*s, *back_n = normals + 3*s;
	  Mesh *mright = xs_back[i]->extrude(back_c, back_t, back_n, num_pts, mid_cap, next_cap);
	  geometry.add_mesh(mright);
	  
	  capped = next_cap;
	  geometry.add_range(i);
	}
    }
}

extern "C" int parse_rxsection_array(PyObject *arg, void *xsvector)
{
  RibbonXSections *xsv = static_cast<RibbonXSections *>(xsvector);
  if (!PySequence_Check(arg))
    {
      PyErr_SetString(PyExc_TypeError,
		      "parse_xsection(): Require list or tuple of RibbonXSection");
      return 0;
    }
  int n = PySequence_Size(arg);
  for (int i = 0 ; i < n ; ++i)
    {
      PyObject *e = PySequence_GetItem(arg, i);
      RibbonXSection *xs;
      if (!parse_rxsection_pointer(e, &xs))
	{
	  PyErr_SetString(PyExc_TypeError,
			  "parse_xsection(): List element is not a RibbonXSection pointer");
	  return 0;
	}
      xsv->push_back(xs);
    }
  return 1;
}

extern "C" PyObject *
ribbon_extrusions(PyObject *, PyObject *args, PyObject *keywds)
{
  FArray centers, tangents, normals;
  IArray ranges;
  int num_res;
  RibbonXSections xs_front, xs_back;
  Geometry *g;
  const char *kwlist[] = {"centers", "tangents", "normals", "ranges", "num_res",
			  "xs_front", "xs_back", "geometry", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&O&iO&O&O&"),
				   (char **)kwlist,
				   parse_float_n3_array, &centers,
				   parse_float_n3_array, &tangents,
				   parse_float_n3_array, &normals,
				   parse_int_n2_array, &ranges,
				   &num_res,
				   parse_rxsection_array, &xs_front,
				   parse_rxsection_array, &xs_back,
				   parse_geometry_pointer, &g))
    return NULL;

  if (!centers.is_contiguous() || !tangents.is_contiguous() || !normals.is_contiguous() ||
      !ranges.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "ribbon_extrusions(): Centers, tangents, normals and ranges arrays must be contiguous");
      return NULL;
    }
  if (normals.size(0) != centers.size(0) || tangents.size(0) != centers.size(0))
    {
      PyErr_Format(PyExc_TypeError,
		   "ribbon_extrusions(): Centers (%s), tangents (%s), and normals (%s) must have same size",
		   centers.size_string().c_str(), tangents.size_string().c_str(),
		   normals.size_string().c_str());
      return NULL;
    }

  ribbon_extrusions(centers.values(), tangents.values(), normals.values(), centers.size(0),
		    ranges.values(), ranges.size(0), num_res, xs_front, xs_back, *g);
  
  return python_none();
}

static bool ribbon_vertex_colors(Residue **residues, int nres, int *triangle_ranges, int nranges,
                                 unsigned char *colors, int ncolors)
{
  for (int i = 0 ; i < nranges ; ++i, triangle_ranges += 5)
    {
      int ri = triangle_ranges[0], v_start = triangle_ranges[3], v_end = triangle_ranges[4];
      if (ri >= nres)
        {
          PyErr_Format(PyExc_ValueError,
                       "ribbon_vertex_colors(): Residue number %d out of range (%d)", ri, nres);
          return false;
        }
      if (v_end > ncolors)
        {
          PyErr_Format(PyExc_ValueError,
                       "ribbon_vertex_colors(): Vertex number %d out of range (%d)", v_end, ncolors);
          return false;
        }
      const atomstruct::Rgba &rgba = residues[ri]->ribbon_color();
      unsigned char r = rgba.r, g = rgba.g, b = rgba.b, a = rgba.a;
      for (int v = v_start ; v < v_end ; ++v)
        {
          unsigned char *c = colors + 4*v;
          c[0] = r; c[1] = g; c[2] = b; c[3] = a;
        }
    }
  return true;
}

extern "C" PyObject *
ribbon_vertex_colors(PyObject *, PyObject *args, PyObject *keywds)
{
  Residues res;
  IArray triangle_ranges;
  Reference_Counted_Array::Array<unsigned char> colors;
  const char *kwlist[] = {"residues", "triangle_ranges", "colors", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&O&O&"),
				   (char **)kwlist,
				   parse_residues, &res,
				   parse_int_2d_array, &triangle_ranges,
				   parse_uint8_n4_array, &colors))
    return NULL;

  if (!triangle_ranges.is_contiguous() || !colors.is_contiguous())
    {
      PyErr_SetString(PyExc_TypeError,
		      "ribbon_vertex_colors(): Ranges and color arrays must be contiguous");
      return NULL;
    }

  if (!ribbon_vertex_colors(res.pointers, res.count,
                            triangle_ranges.values(), triangle_ranges.size(0),
                            colors.values(), colors.size(0)))
    return NULL;	// Range error
  
  return python_none();
}
