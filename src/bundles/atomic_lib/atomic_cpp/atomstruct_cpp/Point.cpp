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

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Point.h"
#include <sstream>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
// not defined on Windows
# define M_PI 3.14159265358979323846
#endif

namespace atomstruct {
    
Real
Point::angle(const Point& pt1, const Point& pt3) const
{
    Real d1_0 = pt1[0] - _xyz[0];
    Real d1_1 = pt1[1] - _xyz[1];
    Real d1_2 = pt1[2] - _xyz[2];
    Real d3_0 = pt3[0] - _xyz[0];
    Real d3_1 = pt3[1] - _xyz[1];
    Real d3_2 = pt3[2] - _xyz[2];

    Real dot = d1_0 * d3_0 + d1_1 * d3_1 + d1_2 * d3_2;
    Real d1 = distance(pt1);
    Real d3 = distance(pt3);
    if (d1 <= 0.0 || d3 <= 0.0)
        return 0.0;
    dot /= (d1 * d3);
    if (dot > 1.0)
        dot = 1.0;
    else if (dot < -1.0)
        dot = -1.0;
    return 180.0 * acos(dot) / M_PI;
}

static inline double
row_mul(const double row[4], const Point& crd)
{
    return row[0] * crd[0] + row[1] * crd[1] + row[2] * crd[2] + row[3];
}

Point
Point::mat_mul(const PositionMatrix pos) const
{
    auto x = row_mul(pos[0], *this);
    auto y = row_mul(pos[1], *this);
    auto z = row_mul(pos[2], *this);
    return Point(x, y, z);
}

void
Point::normalize()
{
    auto len = length();
    if (len == 0.0)
        throw std::domain_error("Can't normalize if length is zero");
    _xyz[0] /= len;
    _xyz[1] /= len;
    _xyz[2] /= len;
}

Real
Point::sqdistance(const Point& pt) const
{
    Real q1 = _xyz[0] - pt._xyz[0];
    Real q2 = _xyz[1] - pt._xyz[1];
    Real q3 = _xyz[2] - pt._xyz[2];
    return q1 * q1 + q2 * q2 + q3 * q3;
}

std::string
Point::str() const
{
    std::stringstream crd_string;
    crd_string << "(";
    for (int i = 0; i < 3; ++i) {
        crd_string << _xyz[i];
        if (i < 2)
            crd_string << ", ";
    }
    crd_string << ")";
    return crd_string.str();
}

void
Point::xform(const PositionMatrix pos)
{
    auto x = row_mul(pos[0], *this);
    auto y = row_mul(pos[1], *this);
    auto z = row_mul(pos[2], *this);
    _xyz[0] = x;
    _xyz[1] = y;
    _xyz[2] = z;
}

} //  namespace atomstruct
