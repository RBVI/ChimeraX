#include <iostream>
#include <otf/Array.h>

namespace plugh {

class xyzzy {
public:
	int	mode() const;
	void	setMode(int m);
	void	setMode(const char *name);

	otf::Array<double, 4> coords() const;
	void	setCoords(double s);
	void	setCoords(double s, double t);
	void	setCoords(double s, double t, double u);
	void	setCoords(double s, double t, double u, double v);

	void            wireStipple(/*OUT*/ int *factor, /*OUT*/ int *pattern);
	void            setWireStipple(int factor, int pattern);
};

} // namespace plugh
