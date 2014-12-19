// vi: set expandtab ts=4 sw=4:
#ifndef basegeom_Point
#define basegeom_Point

#include <cmath>
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
    Real  angle(const Point& pt1, const Point& pt3) const;
    static Real  angle(const Point& pt1, const Point& pt2, const Point& pt3) {
        return pt2.angle(pt1, pt3);
    }
    Real  distance(const Point& pt) const { return std::sqrt(sqdistance(pt)); }
    static Real  distance(const Point& pt1, const Point& pt2) {
            return pt1.distance(pt2);
    }
    void  set_xyz(Real x, Real y, Real z) {
        _xyz[0] = x; _xyz[1] = y; _xyz[2] = z;
    }
    Real  sqdistance(const Point  &pt) const;
    Real&  operator[](int index) { return _xyz[index]; }
    const Real&  operator[](int index) const { return _xyz[index]; }
    std::string  str() const;
};

} //  namespace basegeom

#endif  // basegeom_Point
