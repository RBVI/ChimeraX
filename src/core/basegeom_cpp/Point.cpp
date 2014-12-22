// vi: set expandtab ts=4 sw=4:
#include "Point.h"
#include <sstream>
#include <cmath>

namespace basegeom {
    
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

} //  namespace basegeom
