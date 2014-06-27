#include <iostream>

namespace plugh {

class xyzzy {
public:
	enum Mode { Read, Write, Append };
	int	readfile(/*NULL_OK*/ const char *filename);
	int	writefile(const char *filename = NULL);
	int	readfile(std::istream &is);
	int	writefile(std::ostream &os);
	int	readfile2(std::istream &is, Mode mode);
	int	writefile2(std::ostream &os, Mode mode);
	void	setMode(Mode mode);
	void	setOther(const xyzzy *ref = NULL);
};

} // namespace plugh
