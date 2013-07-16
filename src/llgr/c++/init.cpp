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
	glewExperimental = true;	// Core profile workaround for glew 1.9
	GLenum err = glewInit();
	if (err == GLEW_OK) {
		err = glGetError();	// eat INVALID_ENUM error
#ifdef __APPLE__
		// compiling a shader program fails unless a VAO is bound
		static GLuint vao = glGenVertexArrays(1);
		glBindVertexArray(vao);
#endif
		initialized = true;
		return;
	}
	std::cerr << "glewInit error: " << reinterpret_cast<const char *>(glewGetErrorString(err)) << '\n';
	throw std::runtime_error("unable to initialize glew library");
}

} // namespace llgr
