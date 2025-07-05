# Day 1

## What I learned?

Multidimensional data, simple implementation of matrix multiplication.

## In my own words

We know that to invoke a kernel function, we supply the execution configuration parameters (these are the values between `<<<` and `>>>`). The first value is the `gridDim`, or the grid dimension. In short, it is how many blocks are in a grid, hence the use of the word dimension. The second value is the `blockDim` or the block dimension. In short it is how many threads are in a block. Note they are 3 dimensional. In order to make them 2 or 1 dimensional, we set the additional unused parameters to 1.

Each thread execute the same kernel function. In order to know which data each thread is working with, we can use `blockIdx`, `blockDim`, and `gridDim`.

The matrix multiplication implementation is pretty straightforward, we're calculating the dot product the first matrix's rows and the second matrix's columns. We supply the kernel with 1 block and the resulting matrix's dimension as the thread count. That way, each thread is computing each element of the resulting matrix.
