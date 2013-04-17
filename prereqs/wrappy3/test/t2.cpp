#include "t2.h"

namespace plugh {

#if 0
struct xyzzy: otf::WrapPy<xyzzy> {
	// FINAL
	// EMBED
	int	zero();
	int	one(float d);
	int	two(int i, float d);
};
#endif


int
xyzzy::i() const
{
	return int(d);
}

void
xyzzy::setI(int i)
{
	d = i;
}

xyzzy *
xyzzy::obj() const
{
	return NULL;
}

void
xyzzy::setObj(xyzzy *)
{
}

int
xyzzy::zero()
{
	return 0;
}

int
xyzzy::one(float d)
{
	this->d = d;
	return 1;
}

int
xyzzy::two(int i, float d)
{
	this->d = i * d;
	return 2;
}

void
xyzzy::array(/*OUT*/ float rgba[4])
{
	rgba[0] = rgba[1] = rgba[2] = rgba[3] = 1;
}

void
xyzzy::output(/*OUT*/ xyzzy& out)
{
	static xyzzy tmp(1);
	out = tmp;
}

int
foo(xyzzy*)
{
	return 0;
}

} // end namespace plugh
