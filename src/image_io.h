#ifndef __IMAGE_IO_H__
#define __IMAGE_IO_H__

int load_greyscale_png(const char *filename, unsigned char **image_data, int *width, int *height);
int write_png(const char *filename, unsigned char *buffer, int width, int height, int bit_depth);

#endif
