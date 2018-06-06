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

#ifndef atomstruct_ribbonxsection
#define atomstruct_ribbonxsection

#include <numpy/arrayobject.h>

#include "imex.h"
#include <arrays/pythonarray.h>

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace atomstruct {

class ATOMSTRUCT_IMEX RibbonXSection {
private:
    void _generate_normals();
    PyObject* _extrude_smooth(const FArray& centers, const FArray& tangents,
                              const FArray& normals, const CArray& color,
                              bool cap_front, bool cap_back, int offset) const;
    PyObject* _extrude_faceted(const FArray& centers, const FArray& tangents,
                               const FArray& normals, const CArray& color,
                               bool cap_front, bool cap_back, int offset) const;
    PyObject* _blend_smooth(const IArray& back_band,
                            const IArray& front_band) const;
    PyObject* _blend_faceted(const IArray& back_band,
                             const IArray& front_band) const;
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

    PyObject* extrude(const FArray& centers, const FArray& tangents,
                      const FArray& normals, const CArray& color,
                      bool cap_front, bool cap_back, int offset) const;
    PyObject* blend(const IArray& back_band, const IArray& front_band) const;
    void* scale(float x_scale, float y_scale) const;
    void* arrow(float x1_scale, float y1_scale, float x2_scale, float y2_scale) const;
};

}  // namespace atomstruct

#endif  // atomstruct_ribbonxsection
