#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#include "image_io.h"

// Load 8-bit PNG data into a new buffer
int load_greyscale_png(const char *filename, unsigned char **image_data, int *width, int *height) {
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		perror("Error opening file");
		return -1;
	}

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) {
		fclose(fp);
		fprintf(stderr, "Error creating png read struct\n");
		return -1;
	}

	png_infop info = png_create_info_struct(png);
	if (!info) {
		png_destroy_read_struct(&png, NULL, NULL);
		fclose(fp);
		fprintf(stderr, "Error creating png info struct\n");
		return -1;
	}

	if (setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &info, NULL);
		fclose(fp);
		fprintf(stderr, "Error during PNG reading\n");
		return -1;
	}

	png_init_io(png, fp);
	png_read_info(png, info);

	*width = png_get_image_width(png, info);
	*height = png_get_image_height(png, info);
	png_byte color_type = png_get_color_type(png, info);
	png_byte bit_depth = png_get_bit_depth(png, info);

	// Check if the PNG is 8-bit greyscale
	if (color_type != PNG_COLOR_TYPE_GRAY || bit_depth != 8) {
		fprintf(stderr, "Error: PNG file is not 8-bit greyscale\n");
		png_destroy_read_struct(&png, &info, NULL);
		fclose(fp);
		return -1;
	}

	// Allocate memory for image data
	*image_data = (unsigned char *)malloc((*width) * (*height));
	if (!*image_data) {
		fprintf(stderr, "Error allocating memory for image data\n");
		png_destroy_read_struct(&png, &info, NULL);
		fclose(fp);
		return -1;
	}

	// Read image data row by row
	png_bytep rows[*height];
	for (int y = 0; y < *height; y++) {
		rows[y] = (*image_data) + y * (*width);
	}
	png_read_image(png, rows);

	png_destroy_read_struct(&png, &info, NULL);
	fclose(fp);

	return 0;  // Success
}

// Write 8 or 16 bit greyscale data to a PNG file
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
