// vim: set expandtab ts=4 sw=4:
#ifndef basegeom_Point
#define basegeom_Point

#include <string>
#include "imex.h"
#include "Real.h"

namespace basegeom {
    
class BASEGEOM_IMEX Point {
    Real  _xyz[3];
public:
    Point(Real x, Real y, Real z) {
        _xyz[0] = x; _xyz[1] = y; _xyz[2] = z;
    }
    Point(Real *xyz) {
        for (int i=0; i<3; ++i) _xyz[i] = *xyz++;
    }
    Point() {
        _xyz[0] = _xyz[1] = _xyz[2] = 0.0;
    }
    virtual  ~Point() {}
    Real  sqdistance(const Point &pt) const;
    void  set_xyz(Real x, Real y, Real z) {
        _xyz[0] = x; _xyz[1] = y; _xyz[2] = z;
    }
    Real&  operator[](int index) { return _xyz[index]; }
    const Real&  operator[](int index) const { return _xyz[index]; }
    std::string  str() const;
};

} //  namespace basegeom

#endif  // basegeom_Point
