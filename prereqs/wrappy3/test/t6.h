#include <iostream>
#include <otf/Array.h>

namespace plugh {

	// xyzzy is like a Vector

class xyzzy {
	// NUMBER METHODS
	// SEQUENCE METHODS
	// FINAL
public:
	xyzzy() {}
	//xyzzy(otf::Array<double, 3> v3);
	xyzzy(double v3[3]);
	xyzzy(double x, double y, double z);
	double	x() const;
	void	setX(double x);
	double	y() const;
	void	setY(double y);
	double	z() const;
	void	setZ(double z);
//	const otf::Array<double, 3> data() const;
	bool	operator==(const xyzzy &r) const;
	bool	operator!=(const xyzzy &r) const;
	double	&operator[](int index);
	const double	&operator[](int index) const;
	xyzzy	operator+(const xyzzy &r) const;
	void	operator+=(const xyzzy &r);
	xyzzy	operator-() const;
	xyzzy	operator-(const xyzzy &r) const;
	void	operator-=(const xyzzy &r);
	double	operator*(const xyzzy &r) const;
	xyzzy	operator*(double f) const;
	xyzzy	operator*(int i) const;
	xyzzy	operator/(double f) const;
	void	operator*=(double f);
	void	operator/=(double f);
	bool	operator!() const;
	double	sqlength() const;
	double	length() const;
	void	normalize();
	void	setLength(double newlen);
	void	negate();
	int	size() const { return 3; }
};

extern xyzzy operator+(const xyzzy &r);

extern xyzzy operator*(double f, const xyzzy &r);

extern std::ostream& operator<<(std::ostream& os, const xyzzy& x);

} // namespace plugh
