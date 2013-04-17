#include <iostream>

namespace plugh {

class xyzzy {
	// NUMBER METHODS
public:
	double	operator*(const xyzzy &r) const;
	xyzzy	operator*(double f) const;
};

extern xyzzy operator*(double f, const xyzzy &r);

} // namespace plugh
