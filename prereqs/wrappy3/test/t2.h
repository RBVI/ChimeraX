#ifndef t2_h
# define t2_h

#include <otf/WrapPy2.h>

namespace plugh {

struct xyzzy {
	// FINAL
	// EMBED
	// IMPLICIT COPY CONSTRUCTOR
	xyzzy(): c(0), d(0), f(1) {}
	xyzzy(double x): c(0), d(x), f(0) {}
	int	i() const;
	void	setI(int i);
	xyzzy	*c;
	// READONLY: d
	double	d;
	// WRITEONLY: f
	float	f;
	xyzzy	*obj() const;
	void	setObj(xyzzy *o);
	// ALLOW THREADS: zero
	int	zero();
	int	one(float d);
	int	two(int i, float d);
	static const int HHGTTG = 42;
	void	array(/*OUT*/ float rgba[4]);
	//void	matrix(/*OUT*/ float xform[4][4]);
	void	output(/*OUT*/ xyzzy& out);
};

extern int foo(xyzzy* xy);

} // end namespace plugh

#endif
