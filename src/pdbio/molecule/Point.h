#ifndef molecule_Point
#define molecule_Point

#include <string>
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
	Real sqdistance(const Point &pt) const;
	void set_xyz(Real x, Real y, Real z) {
		_xyz[0] = x; _xyz[1] = y; _xyz[2] = z;
	}
	Real &operator[](int index) { return _xyz[index]; }
	const Real &operator[](int index) const { return _xyz[index]; }
	std::string str() const;
};

#endif  // molecule_Point
