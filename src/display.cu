#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>


#include <cuda_gl_interop.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
//#include "gpu_funcs.h"
//#include "gpu_util.h"
#include "display.h"

// Shared resources
GLuint pbo, tex;
cudaGraphicsResource *cuda_pbo_resource;

SharedParameters_t parameters = {0.0f, false};
volatile bool running = true;

extern pthread_t computation_thread;
extern pthread_mutex_t param_mutex;

// OpenGL display function
void display() {
	glClear(GL_COLOR_BUFFER_BIT);

	// Bind texture to PBO
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Render texture
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(-1, -1);
	glTexCoord2f(1, 0); glVertex2f(1, -1);
	glTexCoord2f(1, 1); glVertex2f(1, 1);
	glTexCoord2f(0, 1); glVertex2f(-1, 1);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();
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
void cleanup() {
	printf("cleanup start\n");

	running = false;
	pthread_join(computation_thread, NULL);

	cudaGraphicsUnregisterResource(cuda_pbo_resource);
	glDeleteBuffers(1, &pbo);
	glDeleteTextures(1, &tex);

	pthread_mutex_destroy(&param_mutex);

	printf("cleanup done\n");
}

// Initialize OpenGL and CUDA interop
void initGL(int *argc, char **argv, int width, int height) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA OpenGL Interop with pthreads");
	glewInit();

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Register PBO with CUDA
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to register PBO with CUDA: %s\n", cudaGetErrorString(err));
    exit(1);
}

	// Register GLUT callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutCloseFunc(cleanup);
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
	const int width = 512, height = 512;
	float local_param = 0.0f;

	printf("computationThreadFunc started\n");

	while (running) {
		uchar4 *d_data;
		size_t num_bytes;

		// Map PBO for CUDA access
		err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to map PBO resource: %s\n", cudaGetErrorString(err));
    exit(1);
}

		cudaGraphicsResourceGetMappedPointer((void **)&d_data, &num_bytes, cuda_pbo_resource);

		// Lock parameter access
		pthread_mutex_lock(&param_mutex);
		if (parameters.update_needed) {
			local_param = parameters.some_parameter;
			parameters.update_needed = false;
		}
		pthread_mutex_unlock(&param_mutex);

		// Run the CUDA kernel
		dim3 block(16, 16);
		dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
		//optimizationKernel<<<grid, block>>>(d_data, width, height, local_param);
err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
}

		// Unmap PBO
		cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

		// Yield to other threads
		usleep(10000); // Small delay to avoid 100% CPU usage
	}

	printf("computationThreadFunc stopped\n");

	return NULL;
}
