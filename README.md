# string-art-cuda #
String art generator using simulated annealing. GPU accelerated using CUDA.

## Description ##

There are several string art generators available, however they use greedy algorithms to select lines sequentially.
This method works quite well, but it may be possible to generate better images with a a non-greedy algorithm that optimises globally.

The difficulty with optimising the whole image each step is that it requires calculating a fitness function for all lines, every step. There will be a very large number of steps required before the image will converge near to a (hopefully) global maximum, so it is critical to minimise the computation time of the fitness function. It is for this reason that it was decided to use GPU acceleration.

The approach will be to generate sequences of string connections on the CPU, using simulated annealing to move strings between nearby nails. The fitness function calculation will run on the GPU, comparing the target image to the lines created by the proposed list of connections.

The project is in a very early stage - more information will be added as development progresses.

## Requirements ##

* NVIDIA GPU
* Linux install, or VM with GPU access
* Installed NVIDIA CUDA Toolkit
* libpng-dev (install: `apt install libpng-dev`)

## Progress ##

- [x] Data structures for input image, output image, line data
- [x] Initialisation/cleanup for CPU/GPU buffers
- [x] Calculation of pixel coverage by lines of given width
- [x] Connection validation to prevent short/repeated lines
- [x] Basic drawing of many lines on GPU
- [ ] Efficient drawing of many lines on GPU
- [ ] GPU accelerated fitness function calculation
- [ ] Speed optimisation of fitness function (GPU)
- [ ] Simulated annealing algorithm implementation (CPU)
- [ ] Tuning of simulated annealing algorithm
- [ ] Importance map to prioritise areas of the image
- [ ] Multi-scale optimisation?
- [ ] Support arbitrary nail positions (convex shape only)
- [ ] Simplify parameter configuration
- [ ] Support loading of parameters from a file
- [ ] Make it prettier: GUI with progress graphs etc

## Limitations ##

* Currently the nail positions are fixed in a circle (will be changed later)
* It's going to be slow (no idea how slow until basic full functionality is done)
* No useful output until the optimisation stage is working
