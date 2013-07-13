#include "llgr_int.h"
#include <stdexcept>
#include <string>
#include <sstream>

#define COMPAT_GL2

namespace llgr {

bool initialized = false;

void
init()
{
	GLenum err = glewInit();
	if (err == GLEW_OK) {
		initialized = true;
		return;
	}
	std::cerr << "glewInit error: " << reinterpret_cast<const char *>(glewGetErrorString(err)) << '\n';
	throw std::runtime_error("unable to initialize glew library");
}

} // namespace llgr
