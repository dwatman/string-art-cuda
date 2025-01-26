#include <GL/glew.h>
#include <GL/freeglut.h>

#include <stdio.h>
#include <pthread.h>

#include "settings.h"
#include "geometry.h"
#include "gpu_funcs.h"
#include "gpu_util.h"
#include "display.h"
#include "image_io.h"

GLuint texid0=0;
int window0=0;

extern SharedParameters_t parameters;
extern volatile uint8_t running;
extern volatile uint8_t update_image;

extern pthread_t computation_thread;
extern pthread_mutex_t param_mutex;

extern uint8_t *h_imageOut;

void display0(void) {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
	glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer

	// Bind texture
	glBindTexture(GL_TEXTURE_2D, texid0);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DATA_SIZE, DATA_SIZE, GL_LUMINANCE, GL_UNSIGNED_BYTE, h_imageOut);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glBegin(GL_QUADS);
		glColor3f(1, 1, 1);
		glTexCoord2f(0, 0); 	glVertex2f(-1, 1); 	// top left
		glTexCoord2f(1, 0);		glVertex2f( 1, 1);	// top right
		glTexCoord2f(1, 1); 	glVertex2f( 1,-1);	// bottom right
		glTexCoord2f(0, 1);		glVertex2f(-1,-1);	// bottom left
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

	glutSwapBuffers(); // render now
}

void timerEvent(int value) {
	// Draw windows
	glutSetWindow(window0);	glutPostRedisplay();

	// Request copy of image from GPU for next display event
	update_image = 1;

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

// OpenGL keyboard callback to update parameters
void keyboard(unsigned char key, int x, int y) {
	pthread_mutex_lock(&param_mutex);
	if (key == 'a') {
		parameters.acceptThresh *= 0.9;
		if (parameters.acceptThresh < 0.0) parameters.acceptThresh = 0.0;
		parameters.update_needed = 1;
	} else if (key == 'd') {
		parameters.acceptThresh /= 0.9;
		if (parameters.acceptThresh > 1.0) parameters.acceptThresh = 1.0;
		parameters.update_needed = 1;
	}
	if (key == 'z') {
		if (parameters.maxMoveDist > 1) parameters.maxMoveDist -= 1;
		parameters.update_needed = 1;
	} else if (key == 'c') {
		parameters.maxMoveDist += 1;
		parameters.update_needed = 1;
	}

	pthread_mutex_unlock(&param_mutex);
	printf("acceptThresh: %f  maxMoveDist: %i\n", parameters.acceptThresh, parameters.maxMoveDist);
}

// Cleanup function
void GlCleanup() {
	printf("GlCleanup start\n");

	running = 0;
	pthread_join(computation_thread, NULL);

	glDeleteTextures(1, &texid0);

	pthread_mutex_destroy(&param_mutex);

	printf("GlCleanup done\n");
}

// Initialize OpenGL and CUDA interop
void initGL(int *argc, char **argv, int width, int height) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(10, 10);
	glutInitWindowSize(width, height);
	window0 = glutCreateWindow("CUDA String Art Optimiser");
	glewInit();

	// Register GLUT callbacks
	glutDisplayFunc(display0);
	glutKeyboardFunc(keyboard);
	glutCloseFunc(GlCleanup);

	glutSetWindow(window0);
	glGenTextures(1, &texid0);
	glBindTexture(GL_TEXTURE_2D, texid0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, DATA_SIZE, DATA_SIZE, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}
