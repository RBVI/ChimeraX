#include "llgr_int.h"
#include <stdexcept>
#include <string>
#include <sstream>

#define COMPAT_GL2

namespace llgr {

GLuint program_vao;

bool initialized = false;

void
init()
{
	glewExperimental = true;	// Core profile workaround for glew 1.9
	GLenum err = glewInit();
	if (err == GLEW_OK) {
		err = glGetError();	// eat INVALID_ENUM error for OpenGL 3
#if __APPLE__ && __MACH__
		// compiling a shader program fails unless a VAO is bound
		glGenVertexArrays(1, &program_vao);
#endif
		initialized = true;
		return;
	}
	std::cerr << "glewInit error: " << reinterpret_cast<const char *>(glewGetErrorString(err)) << '\n';
	throw std::runtime_error("unable to initialize glew library");
}

} // namespace llgr
