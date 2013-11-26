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
	// TODO: use shader uniforms to set lighting
	static GLfloat zero[4] = { 0, 0, 0, 0 };
	static GLfloat one[4] = { 1, 1, 1, 1 };
	static GLfloat ambient[4] = { 0.197f, 0.197f, 0.197f, 1 };
	static GLfloat diffuse0[4] = { 0.432f, 0.432f, 0.432f, 1 };
	static GLfloat position0[4] = { 0.251f, 0.251f, 0.935f, 0 };
	static GLfloat diffuse1[4] = { 0.746f, 0.746f, 0.746f, 1 };
	static GLfloat position1[4] = { -0.357f, 0.66f, 0.66f, 0 };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse1);
	glLightfv(GL_LIGHT1, GL_SPECULAR, one);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, zero);

	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, one);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30);
	//glDisable(GL_CULL_FACE);	// debugging

	//glClearColor(.5f, .2f, .2f, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	llgr::render();
	glutSwapBuffers();
}

int
main(int argc, char **argv)
{
	glutInit(&argc, argv);

	//glutInitContextVersion(3,3);
	//glutInitContextProfile(GLUT_CORE_PROFILE);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
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
