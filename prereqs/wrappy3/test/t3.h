#include <otf/WrapPy2.h>

namespace plugh {

class xyzzy: public otf::WrapPyObj {
	xyzzy();
	~xyzzy();
	virtual PyObject* wpyNew() const;
public:
	int	i();
};

} // namespace plugh
