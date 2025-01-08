#include <GL/glew.h>
#include <GL/freeglut.h>

#include <stdio.h>
#include <stdbool.h>
#include <unistd.h> // for usleep
#include <pthread.h>

#include "settings.h"
//#include "gpu_funcs.h"
//#include "gpu_util.h"
#include "display.h"

GLuint texid0=0;
int window0=0;

SharedParameters_t parameters = {0.0f, false};
volatile bool running = true;

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

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

// OpenGL keyboard callback to update parameters
void keyboard(unsigned char key, int x, int y) {
	pthread_mutex_lock(&param_mutex);
	if (key == 'a') {
		parameters.some_parameter += 0.1f;  // Example modification
		parameters.update_needed = true;
	} else if (key == 'd') {
		parameters.some_parameter -= 0.1f;  // Example modification
		parameters.update_needed = true;
	}
	pthread_mutex_unlock(&param_mutex);
	printf("Parameter updated: %f\n", parameters.some_parameter);
}

// Cleanup function
void GlCleanup() {
	printf("GlCleanup start\n");

	running = false;
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
	window0 = glutCreateWindow("CUDA OpenGL Interop with pthreads");
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

// CUDA kernel to simulate optimization work
__global__ void optimizationKernel(uchar4 *data, int width, int height, float param) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		int idx = y * width + x;
		data[idx] = make_uchar4((x + (int)(param * 10)) % 256, y % 256, 128, 255);
	}
}

// CUDA computation thread function
void *computationThreadFunc(void *arg) {
	cudaError_t err;
	float local_param = 0.0f;

	printf("computationThreadFunc started\n");

	while (running) {
		// Lock parameter access
		pthread_mutex_lock(&param_mutex);
		if (parameters.update_needed) {
			local_param = parameters.some_parameter;
			parameters.update_needed = false;
		}
		pthread_mutex_unlock(&param_mutex);

		// Run the CUDA kernel


		// Yield to other threads
		usleep(10000); // Small delay to avoid 100% CPU usage
	}

	printf("computationThreadFunc stopped\n");

	return NULL;
}
