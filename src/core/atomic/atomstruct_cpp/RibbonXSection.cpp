// vi: set expandtab ts=4 sw=4:

#include <iostream>
#include <algorithm>

#include "RibbonXSection.h"

namespace atomstruct {

static void
dump_farray(const char *label, const FArray& v, int nc)
{
    std::cerr << label << "(" << v.size(0) << ", " << nc << "):\n";
    int index[2];
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

static void
dump_farray_addresses(const char *label, const FArray& v)
{
    const float *data = v.values();
    const float *end_data = data + v.size();
    std::cerr << label << ' ' << &v << ' ' << data << ' ' << end_data << '\n';
}

static void
dump_pyarray_addresses(const char *label, PyObject* o)
{
    std::cerr << label << ' ' << o << std::flush;
    if (!PyArray_Check(o)) {
        std::cerr << " not an array object\n";
        return;
    }
    std::cerr << " checked";
    PyArrayObject* a = (PyArrayObject*) o;
    std::cerr << " casted";
    const char* data = PyArray_BYTES(a);
    std::cerr << ' ' << (void*) data << std::flush;
    const char* end_data = data + PyArray_SIZE(a) * PyArray_ITEMSIZE(a);
    std::cerr << ' ' << (void*) end_data << '\n';
}

RibbonXSection::RibbonXSection(FArray* coords, FArray* coords2,
                               FArray* normals, FArray* normals2,
                               bool faceted, FArray* tess)
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
    if (coords == NULL)
        throw std::invalid_argument("no ribbon cross section coordinates");
    if (coords->dimension() != 2 || coords->size(1) != 2)
        throw std::invalid_argument("bad ribbon cross section dimensions");
// std::cerr << "RibbonXSection\n";
// dump_farray("coords", *coords, 2);
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
// dump_farray("xs_normals", xs_normals, 2);
    if (tess)
        tessellation = *tess;
    else
        _tessellate();
}

RibbonXSection::~RibbonXSection()
{
    return;
}

PyObject*
RibbonXSection::extrude(const FArray& centers, const FArray& tangents,
                        const FArray& normals, const FArray& color,
                        bool cap_front, bool cap_back, int offset) const
{
    if (is_faceted)
        return _extrude_faceted(centers, tangents, normals, color,
                                cap_front, cap_back, offset);
    else
        return _extrude_smooth(centers, tangents, normals, color,
                               cap_front, cap_back, offset);
}

PyObject*
RibbonXSection::blend(const IArray& back_band, const IArray& front_band) const
{
    if (is_faceted)
        return _blend_faceted(back_band, front_band);
    else
        return _blend_smooth(back_band, front_band);
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
    int num_coords = xs_coords.size(0);
    int index0[2] = { 0, 0 };
    int index1[2] = { 0, 1 };
    if (!is_faceted) {
        xs_normals = FArray(xs_coords.dimension(), xs_coords.sizes());
        for (int i = 0; i != num_coords; ++i) {
            index0[0] = index1[0] = i;
            float ix = xs_coords.value(index0);
            float iy = xs_coords.value(index1);
            int j = (i + 1) % num_coords;
            index0[0] = index1[0] = j;
            float jx = xs_coords.value(index0);
            float jy = xs_coords.value(index1);
            int k = (i + 2) % num_coords;
            index0[0] = index1[0] = k;
            float kx = xs_coords.value(index0);
            float ky = xs_coords.value(index1);
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
        xs_normals = FArray(xs_coords.dimension(), xs_coords.sizes());
        xs_normals2 = FArray(xs_coords.dimension(), xs_coords.sizes());
        for (int i = 0; i != num_coords; ++i) {
            index0[0] = index1[0] = i;
            float ix = xs_coords.value(index0);
            float iy = xs_coords.value(index1);
            int j = (i + 1) % num_coords;
            index0[0] = index1[0] = j;
            float jx = xs_coords.value(index0);
            float jy = xs_coords.value(index1);
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

static PyObject*
extrude_values(PyObject* vertices, PyObject* normals, PyObject* triangles,
               PyObject* colors, PyObject* front_band, PyObject* back_band)
{
    PyObject* answer = PyTuple_New(6);
    PyTuple_SetItem(answer, 0, vertices);
    PyTuple_SetItem(answer, 1, normals);
    PyTuple_SetItem(answer, 2, triangles);
    PyTuple_SetItem(answer, 3, colors);
    PyTuple_SetItem(answer, 4, front_band);
    PyTuple_SetItem(answer, 5, back_band);
    return answer;
}

PyObject*
RibbonXSection::_extrude_smooth(const FArray& centers, const FArray& tangents,
                                const FArray& normals, const FArray& color,
                                bool cap_front, bool cap_back, int offset) const
{
// std::cerr << "extrude_smooth\n";
// dump_farray("centers:", centers, 3);
// dump_farray("tangents:", tangents, 3);
// dump_farray("normals:", normals, 3);
    // Calculate result sizes
    int num_splines = xs_coords.size(0);
    int num_pts_per_spline = centers.size(0);
    int num_vertices = num_splines * num_pts_per_spline;
    if (cap_front)
        num_vertices += num_splines;
    if (cap_back)
        num_vertices += num_splines;
    // Allocate space for colors
    float *ca_data = NULL;
    PyObject* ca = python_float_array(num_vertices, 4, &ca_data);
    if (!ca)
        return NULL;
    // Repeat color for all vertices
    float *cp = color.values();
    for (int i = 0; i != num_vertices; ++i) {
        float *cai = ca_data + i * 4;
        for (int j = 0; j != 4; ++j)
            cai[j] = cp[j];
    }
    // Allocate space for vertices and normals
    float *va_data = NULL;
    PyObject *va = python_float_array(num_vertices, 3, &va_data);
    float *na_data = NULL;
    PyObject *na = python_float_array(num_vertices, 3, &na_data);
    if (!va || !na)
        return NULL;
    int vindex = 0;
    // Compute binormals as cross product of tangents and normals
    FArray binormals(normals.dimension(), normals.sizes());
    float* binormals_values = binormals.values();
    float* tangents_values = tangents.values();
    float* normals_values = normals.values();
    float* centers_values = centers.values();
    for (int i = 0; i != num_pts_per_spline; ++i) {
        float* bv = binormals_values + i * 3;
        float* tv = tangents_values + i * 3;
        float* nv = normals_values + i * 3;
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
            arrow_length = num_pts_per_spline / 2;
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
            float* center = centers_values + i * 3;
            float* normal = normals_values + i * 3;
            float* binormal = binormals_values + i * 3;
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
    // Allocate space for triangles
    int num_triangles = num_splines * (num_pts_per_spline - 1) * 2;
    if (cap_front)
        num_triangles += tessellation.size(0);
    if (cap_back)
        num_triangles += tessellation.size(0);
    int *ta_data = NULL;
    PyObject* ta = python_int_array(num_triangles, 3, &ta_data);
    if (!ta)
        return NULL;
    // Allocate space for front and back bands
    int *front_band_data;
    int *back_band_data;
    PyObject* front_band = python_int_array(num_splines, &front_band_data);
    PyObject* back_band = python_int_array(num_splines, &back_band_data);
    if (!front_band || !back_band)
        return NULL;
    int tindex = 0;
    // Create triangles
    for (int s = 0; s != num_splines; ++s) {
        int i_start = s * num_pts_per_spline + offset;
        front_band_data[s] = i_start;
        back_band_data[s] = i_start + num_pts_per_spline - 1;
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
        float* tangent = tangents_values;
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
        float* tangent = tangents_values + (num_pts_per_spline - 1) * 3;;
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
    return extrude_values(va, na, ta, ca, front_band, back_band);
}

PyObject*
RibbonXSection::_extrude_faceted(const FArray& centers, const FArray& tangents,
                                 const FArray& normals, const FArray& color,
                                 bool cap_front, bool cap_back, int offset) const
{
// std::cerr << "extrude_faceted\n";
// dump_farray("centers:", centers, 3);
// dump_farray("tangents:", tangents, 3);
// dump_farray("normals:", normals, 3);
    // Calculate result sizes
    int num_splines = xs_coords.size(0);
    int num_pts_per_spline = centers.size(0);
    int num_vertices = num_splines * num_pts_per_spline * 2;
    if (cap_front)
        num_vertices += num_splines;
    if (cap_back)
        num_vertices += num_splines;
    // Allocate space for colors
    float *ca_data = NULL;
    PyObject* ca = python_float_array(num_vertices, 4, &ca_data);
    if (!ca)
        return NULL;
    // Repeat color for all vertices
    float *cp = color.values();
    for (int i = 0; i != num_vertices; ++i) {
        float *cai = ca_data + i * 4;
        for (int j = 0; j != 4; ++j)
            cai[j] = cp[j];
    }
    // Allocate space for vertices and normals
    float *va_data = NULL;
    PyObject *va = python_float_array(num_vertices, 3, &va_data);
    float *na_data = NULL;
    PyObject *na = python_float_array(num_vertices, 3, &na_data);
    if (!va || !na)
        return NULL;
    int vindex = 0;
    // Compute binormals as cross product of tangents and normals
    FArray binormals(normals.dimension(), normals.sizes());
    float* binormals_values = binormals.values();
    float* tangents_values = tangents.values();
    float* normals_values = normals.values();
    float* centers_values = centers.values();
    for (int i = 0; i != num_pts_per_spline; ++i) {
        float* bv = binormals_values + i * 3;
        float* tv = tangents_values + i * 3;
        float* nv = normals_values + i * 3;
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
            arrow_length = num_pts_per_spline / 2; 
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
            float* center = centers_values + i * 3;
            float* normal = normals_values + i * 3;
            float* binormal = binormals_values + i * 3;
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
    // Allocate space for triangles
    int num_triangles = num_splines * (num_pts_per_spline - 1) * 2;
    if (cap_front)
        num_triangles += tessellation.size(0);
    if (cap_back)
        num_triangles += tessellation.size(0);
    int *ta_data = NULL;
    PyObject* ta = python_int_array(num_triangles, 3, &ta_data);
    if (!ta)
        return NULL;
    // Allocate space for front and back bands
    int *front_band_data;
    int *back_band_data;
    PyObject* front_band = python_int_array(num_splines * 2, &front_band_data);
    PyObject* back_band = python_int_array(num_splines * 2, &back_band_data);
    if (!front_band || !back_band)
        return NULL;
    int tindex = 0;
    // Create triangles
    for (int s = 0; s != num_splines; ++s) {
        int i_start = (s * 2) * num_pts_per_spline + offset;
        front_band_data[s * 2] = i_start;
        front_band_data[s * 2 + 1] = i_start + num_pts_per_spline;
        back_band_data[s * 2] = i_start + num_pts_per_spline - 1;
        back_band_data[s * 2 + 1] = i_start + 2 * num_pts_per_spline - 1;
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
        float* tangent = tangents_values;
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
        float* tangent = tangents_values + (num_pts_per_spline - 1) * 3;;
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
    return extrude_values(va, na, ta, ca, front_band, back_band);
}

PyObject*
RibbonXSection::_blend_smooth(const IArray& back_band,
                              const IArray& front_band) const
{
    int size = back_band.size();
    if (front_band.size() != size)
        throw std::logic_error("blending non-identical cross sections");
    int *ta_data = NULL;
    PyObject* ta = python_int_array(size * 2, 3, &ta_data);
    if (!ta)
        return NULL;
    int *back = back_band.values();
    int *front = front_band.values();
    for (int i = 0; i != size; ++i) {
        int j = (i + 1) % size;
        int *t = ta_data + (i * 2) * 3;
        t[0] = back[i];
        t[1] = back[j];
        t[2] = front[i];
        t += 3;
        t[0] = front[i];
        t[1] = back[j];
        t[2] = front[j];
    }
    return ta;
}

PyObject*
RibbonXSection::_blend_faceted(const IArray& back_band,
                               const IArray& front_band) const
{
    int size = back_band.size();
    if (front_band.size() != size)
        throw std::logic_error("blending non-identical cross sections");
    int num_vertices = size / 2;
    int *ta_data = NULL;
    PyObject* ta = python_int_array(num_vertices * 2, 3, &ta_data);
    if (!ta)
        return NULL;
    int *back = back_band.values();
    int *front = front_band.values();
    for (int n = 0; n != num_vertices; ++n) {
        int i = n * 2;
        int j = (i + 3) % size;
        int *t = ta_data + (n * 2) * 3;
        t[0] = back[i];
        t[1] = back[j];
        t[2] = front[i];
        t += 3;
        t[0] = front[i];
        t[1] = back[j];
        t[2] = front[j];
    }
    return ta;
}

void
RibbonXSection::_normalize_normals(FArray& v) const
{
    int size = v.size(0);
    int index0[2] = { 0, 0 };
    int index1[2] = { 0, 1 };
    for (int i = 0; i != size; ++i) {
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
    int num_coords = xs_coords.size(0);
    int size[2] = { num_coords - 2, 3 };
    tessellation = IArray(2, size);
    int* tdata = tessellation.values();
    for (int i = 1; i != num_coords - 1; ++i) {
        tdata[0] = 0;
        tdata[1] = i;
        tdata[2] = i + 1;
        tdata += 3;
    }
}

}  // namespace atomstruct
