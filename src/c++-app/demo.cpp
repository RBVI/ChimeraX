#include <GL/glew.h>
#include <stdio.h>
#include <GL/freeglut.h>
#include <llgr_ui.h>
#include "data.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace llgr;

static bool initialized = false;

static Id shader_id, obj_id, data_id, matrix_id;

void
initialize()
{
	initialized = true;
	char vertex_shader[4096+1], fragment_shader[4096+1];

	std::ifstream input;
	input.open("../shaders/vertexShader.txt");
	input.read(vertex_shader, 4096);
	vertex_shader[input.gcount()] = '\0';
	input.close();
	input.clear();
	input.open("../shaders/fragmentShader.txt");
	input.read(fragment_shader, 4096);
	fragment_shader[input.gcount()] = '\0';
	input.close();

	++shader_id;
	create_program(shader_id, vertex_shader, fragment_shader, NULL);
#if 1
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(projection_matrix);
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(modelview_matrix);
#else
	set_uniform_matrix(0, "ProjectionMatrix", false, llgr::Mat4x4, sizeof projection_matrix, projection_matrix);
	set_uniform_matrix(0, "ModelViewMatrix", false, llgr::Mat4x4, sizeof modelview_matrix, modelview_matrix);
	float normal_matrix[3][3];
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			normal_matrix[i][j] = modelview_matrix[i * 4 + j];
	set_uniform_matrix(0, "NormalMatrix", false, llgr::Mat3x3, sizeof normal_matrix, normal_matrix);
#endif

	for (Sphere *s = &spheres[0]; s != &spheres[sphere_count]; ++s) {
		++data_id;
		create_singleton(data_id, sizeof s->color, s->color);

		AttributeInfos ais;
		AttributeInfo ai("gl_Color", data_id, 0, 0, 4, Float);
		ais.push_back(ai);

		++matrix_id;
		static float mat[4][4] = {
			{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
#if 1
		// default chimera layout
		mat[0][3] = s->center[0];
		mat[1][3] = s->center[1];
		mat[2][3] = s->center[2];
#else
		// OpenGL layout
		mat[3][0] = s->center[0];
		mat[3][1] = s->center[1];
		mat[3][2] = s->center[2];
#endif
		create_matrix(matrix_id, mat, false);

		++obj_id;
		add_sphere(obj_id, s->radius, shader_id, matrix_id, ais);
	}

	for (Cylinder *c = &cylinders[0]; c != &cylinders[cylinder_count]; ++c) {
		++data_id;
		create_singleton(data_id, sizeof c->color, c->color);

		AttributeInfos ais;
		AttributeInfo ai("gl_Color", data_id, 0, 0, 4, Float);
		ais.push_back(ai);

		++matrix_id;
		static float mat[4][4] = {
			{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
		for (unsigned row = 0; row != 3; ++row) {
			for (unsigned col = 0; col != 4; ++col) {
				mat[row][col] = c->orient[row][col];
			}
		}
		create_matrix(matrix_id, mat, false);

		++obj_id;
		add_cylinder(obj_id, c->radius, c->length, shader_id, matrix_id, ais);
	}
}

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
}

void display()
{
	if (!initialized)
		initialize();
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
