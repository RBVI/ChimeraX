#include "Point.h"
#include <sstream>

Real
Point::sqdistance(const Point &pt) const {
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
