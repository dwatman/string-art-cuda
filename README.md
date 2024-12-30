# string-art-cuda #
String art generator using simulated annealing. GPU accelerated using CUDA.

## Requirements ##

* NVIDIA GPU
* Linux install, or VM with GPU access
* Installed NVIDIA CUDA Toolkit
* libpng-dev (install: `apt install libpng-dev`)


## Project Description ##

There are several methods to generate string art, however they have some limitations that this project aims to improve on.
The result should be a higher quality string art image, although the computation time will be longer than other methods.

I figure that if you're going to spend many hours making a string artwork, you might as well spend a bit more processing time at the start generating the best possible line sequence.

### String Art Algorithms ###

#### Greedy Line Selection (sequential) ####
The simplest method is to use a greedy algorithm to select lines sequentially.

Advantages:
* Fast - only a small part of the generated image needs to be evaluated at each step
* Can be further sped up using Radon transform
* Number of lines can be determined automatically by termination criteria

Disadvantages:
* Result is not globally optimal
* Result will change depending the first nail chosen

This method works quite well, but it may be possible to generate better images with a a non-greedy algorithm that optimises globally.

#### Least Squares Optimisation (global) ####

Advantages:
* Globally optimal, in a certain mathematical sense
* Linear algebra libraries are fast

Disadvantages:
* Too much maths - very difficult to make changes without a deep understanding
* Only supports circular shape (in the implementations I have found)
* Requires manual selection of the number of lines to use

#### This Approach (global) ####

Advantages:
* Near to globally optimal, in a visual comparison sense
* Compatible with any optimisation function (simulated annealing used here)
* Simpler to understand and modify
* Supports arbitrary convex shapes (could be extended to any shape with changes to nail selection constraints)
* Supports image weighting to prioritise important areas, such as details of a face vs the background

Disadvantages:
* Full calculation of fitness function required at each step (drawing and comparing all lines)
* Requires manual selection of the number of lines to use
* Result will vary slightly each time it is run
* Slower than other methods

The difficulty with optimising the whole image is that it requires calculating a fitness function using all lines, every step. There will be a very large number of steps required before the image will converge near to a (hopefully) global maximum, so it is critical to minimise the computation time of the fitness function. It is for this reason that it was decided to use GPU acceleration.

The approach will be to generate sequences of string connections on the CPU, using simulated annealing to optimise the sequence. The fitness function calculation will run on the GPU, comparing the target image to the lines created by the proposed list of connections.

The project is in a very early stage - more information will be added as development progresses.

## Progress ##

- [x] Data structures for input image, output image, line data
- [x] Initialisation/cleanup for CPU/GPU buffers
- [x] Calculation of pixel coverage by lines of given width
- [x] Connection validation to prevent short/repeated lines
- [x] Basic drawing of many lines on GPU
- [x] Efficient drawing of many lines on GPU
- [x] GPU accelerated fitness function calculation
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

## Performance Tests ##

NVIDIA GeForce RTX 3080 Ti

300 nails
5000 lines
512x512 output image

DrawLine_kernel (GPU) duration (99.8% of the run time)

* 2024/12/04 Nsight Compute 11.54 ms (initial implementation)
* 2024/12/07 Nsight Compute 10.60 ms (line coverage data in shared memory)
* 2024/12/29 Nsight Compute  5.85 ms (lines in shared memory, processed in batches)
* 2024/12/29 Nsight Compute  1.89 ms (skip unnecessary calculations)
* 2024/12/30 Nsight Compute  1.52 ms (improved memory structure)
