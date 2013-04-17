#include <iostream>

namespace plugh {

class xyzzy {
	// NUMBER METHODS
public:
	double	operator()(const xyzzy &r) const;
	xyzzy	operator()(double f) const;
};

} // namespace plugh
