#include <GL/glew.h>
#include <stdio.h>
#include <GL/freeglut.h>
#include <llgr_ui.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace llgr;

extern int width, height;
extern void initialize();

static bool initialized = false;

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
}

void display()
{
	if (!initialized) {
		initialize();
		initialized = true;
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	llgr::render();
	glutSwapBuffers();
}

int
main(int argc, char **argv)
{
	glutInit(&argc, argv);

	//glutInitContextVersion(3, 3);
	//glutInitContextProfile(GLUT_CORE_PROFILE);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("chimera2 low-level demo");

	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << "unable to initialize glew library\n";
		return 1;
	}

	std::cout << "OpenGL " << glGetString(GL_VERSION)
		<< ", GLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION)
		<< "\n";

	glutReshapeFunc(reshape);
	glutDisplayFunc(display);

	glutMainLoop();
	return 0;
}
