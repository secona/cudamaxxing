# Day 1

## What I learned?

The basics of CUDA, including the program structure, devices vs hosts, kernel functions, memory stuff, simple vector addition.

## In my own words

The structure of a CUDA program is divided into two parts: the host and the devices (notice the plural). The host is the CPU and the devices are GPUs in the computer.

Variables live in either the host and the devices. To make things easier to read, suffixes are used: `_h` for host variables and `_d` for device variables. Similar to C, we have functions to allocate and deallocate memory from the GPU memory, they are `cudaMalloc` and `cudaFree`. We also have a function named `cudaMemcpy` to copy memory data between the host and the device.

CUDA kernels are like functions that run on the GPU. These kernels are noticable by their qualifiers, usually with `__global__` or `__device__`. In addition, we also have a `__host__` qualifier. This qualifier is the default qualifier, meaning all functions without explicit qualifiers, have the `__host__` qualifier.

To invoke a CUDA kernel, we have to specify the number of blocks and the number of threads per block, they appear before the arguments in the `<<<  >>>`. We can launch as many threads as we want, but notice that the amount of threads launched may not be the same as the data. Say you have a vector with n = 100, the number of threads launched may not be equal to 100 and is usually larger than 100. This means we have to "filter" out what threads should compute things and what shouldn't. We definitely don't want to calculate 128 additions when there is only 100 data.

I implemented a simple vector addition. We don't have a for-loop to execute the computation, instead we have "one" CUDA kernel. I like to think of it as calling multiple CUDA kernels at once in parallel. Hence, we have the absence of a loop. The "guard" `i < n` is also what I talked about in the previous paragraph. Notice that when n = 100, the value of `i` may be greater then that. Do a little math and figure out why *that* is the value of `i` :)

- `threadIdx` --- In the current block, what thread index are we?
- `blockDim` --- How many threads do we have in a single block.
- `blockIdx` --- In all `blockDim` blocks, what block index are we?
