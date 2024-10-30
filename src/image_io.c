#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#include "image_io.h"

int write_png(const char *filename, unsigned char *buffer, int width, int height, int bit_depth) {
	FILE *fp = fopen(filename, "wb");
	if (!fp) {
		perror("Could not open file for writing");
		return -1;
	}

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) {
		fprintf(stderr, "Could not create PNG write structure\n");
		fclose(fp);
		return -1;
	}

	png_infop info = png_create_info_struct(png);
	if (!info) {
		fprintf(stderr, "Could not create PNG info structure\n");
		png_destroy_write_struct(&png, NULL);
		fclose(fp);
		return -1;
	}

	if (setjmp(png_jmpbuf(png))) {
		fprintf(stderr, "Error during PNG creation\n");
		png_destroy_write_struct(&png, &info);
		fclose(fp);
		return -1;
	}

	png_init_io(png, fp);

	// Set the image attributes
	png_set_IHDR(png, info, width, height, bit_depth, PNG_COLOR_TYPE_GRAY,
				 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png, info);

	// Allocate row pointers
	png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
	if (bit_depth == 8) {
		for (int y = 0; y < height; y++) {
			row_pointers[y] = buffer + y * width;
		}
	} else if (bit_depth == 16) {
		for (int y = 0; y < height; y++) {
			row_pointers[y] = buffer + y * width * 2;
		}
	} else {
		fprintf(stderr, "Unsupported bit depth: %d\n", bit_depth);
		png_destroy_write_struct(&png, &info);
		free(row_pointers);
		fclose(fp);
		return -1;
	}

	// Write image data
	png_write_image(png, row_pointers);
	png_write_end(png, NULL);

	// Clean up
	free(row_pointers);
	png_destroy_write_struct(&png, &info);
	fclose(fp);

	return 0;
}
