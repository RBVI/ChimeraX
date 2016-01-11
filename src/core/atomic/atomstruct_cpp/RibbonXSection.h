// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_ribbonxsection
#define atomstruct_ribbonxsection

#include <numpy/arrayobject.h>

#include "imex.h"
#include "pythonarray.h"

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
                              const FArray& normals, const FArray& color,
                              bool cap_front, bool cap_back, int offset) const;
    PyObject* _extrude_faceted(const FArray& centers, const FArray& tangents,
                               const FArray& normals, const FArray& color,
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
                   FArray* tess = NULL);
    virtual ~RibbonXSection();

    PyObject* extrude(const FArray& centers, const FArray& tangents,
                      const FArray& normals, const FArray& color,
                      bool cap_front, bool cap_back, int offset) const;
    PyObject* blend(const IArray& back_band, const IArray& front_band) const;
};

}  // namespace atomstruct

#endif  // atomstruct_ribbonxsection
