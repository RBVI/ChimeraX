#include <stdexcept>
#include <vector>

namespace plugh {

class xyzzy;

class xyzzy {
public:
//	void    setFunc(int i) throw ();
//	int     func() throw ();
	bool	operator<(const xyzzy &t) const throw (std::logic_error);
	bool	operator<(int r) const throw (std::logic_error);
	typedef std::vector<char> cvec;
	std::pair<cvec::iterator, cvec::iterator> pairfunc();
	typedef std::pair<cvec::const_iterator, cvec::const_iterator>
								CVecAtoms;
	CVecAtoms   traverseAtoms(void *root);
};

bool operator<(int i, const xyzzy &t) throw (std::logic_error);

class xyzzy;

} // namespace plugh
