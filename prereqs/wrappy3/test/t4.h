#include <string>
#include <utility>
#include <vector>

namespace plugh {

class xyzzy {
public:
	std::pair<std::string, std::vector<xyzzy *> > func();
};

} // namespace plugh
