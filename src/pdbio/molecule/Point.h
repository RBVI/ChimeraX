#ifndef molecule_Point
#define molecule_Point

#include "Real.h"

class Point {
	Real	_xyz[3];
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
	Real sqdistance(const Point &pt) const {
		return _xyz[0]*pt._xyz[0] + _xyz[1]*pt._xyz[1] + _xyz[1]*pt._xyz[2];
	}
	void set_xyz(Real x, Real y, Real z) {
		_xyz[0] = x; _xyz[1] = y; _xyz[2] = z;
	}
};

#endif  // molecule_Point
