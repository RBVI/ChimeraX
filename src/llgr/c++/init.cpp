// vi: set expandtab shiftwidth=4 softtabstop=4:
#include "llgr.h"
#include "llgr_int.h"
#include <string>
#include <sstream>

#define COMPAT_GL2

namespace llgr {

namespace internal {
GLuint program_vao;

bool initialized = false;

void
init()
{
	glewExperimental = true;	// Core profile workaround for glew 1.9
	GLenum err = glewInit();
	if (err == GLEW_OK) {
		err = glGetError();	// eat INVALID_ENUM error for OpenGL 3
		initialized = true;
		return;
	}
	std::cerr << "glewInit error: " << reinterpret_cast<const char *>(glewGetErrorString(err)) << '\n';
	throw std::runtime_error("unable to initialize glew library");
}

} // namespace internal

} // namespace llgr
