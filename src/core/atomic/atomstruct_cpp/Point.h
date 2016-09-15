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

#ifndef atomstruct_Point
#define atomstruct_Point

#include <cmath>
#include <iostream>
#include <string>

#include "imex.h"
#include "Real.h"

namespace atomstruct {
    
class ATOMSTRUCT_IMEX Point {
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
    Real  length() const {
        return sqrt(_xyz[0] * _xyz[0] + _xyz[1] * _xyz[1] + _xyz[2] * _xyz[2]);
    }
    void  normalize();
    void  set_xyz(Real x, Real y, Real z) {
        _xyz[0] = x; _xyz[1] = y; _xyz[2] = z;
    }
    Real  sqdistance(const Point  &pt) const;
    Point  operator*(Real mul) const {
            return Point(_xyz[0] * mul, _xyz[1] * mul, _xyz[2] * mul);
    }
    Point  operator+(const Point  &pt) const {
        return Point(_xyz[0] + pt._xyz[0], _xyz[1] + pt._xyz[1], _xyz[2] + pt._xyz[2]);
    }
    Point  operator-(const Point  &pt) const {
        return Point(_xyz[0] - pt._xyz[0], _xyz[1] - pt._xyz[1], _xyz[2] - pt._xyz[2]);
    }
    Real&  operator[](int index) { return _xyz[index]; }
    const Real&  operator[](int index) const { return _xyz[index]; }
    std::string  str() const;
};

inline std::ostream&
operator<<(std::ostream& os, const Point& p) {
    os << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
    return os;
}

} //  namespace atomstruct

#endif  // atomstruct_Point
