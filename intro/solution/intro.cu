/*
 * Solution to the introductory CUDA exercise
 */

/*
 * A simple CUDA exercise that negates an array of floats.
 * Introduces device memory management and kernel invocation.
 */

#include <iostream>
#include <cstdlib> // for malloc & free

// Error checking function
#define myCudaCheck(result) { cudaErrorCheck((result), __FILE__, __LINE__); }
inline void cudaErrorCheck(cudaError_t err, const char* file, int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
    exit(err);
  }
}


// The number of elements in the array
const int size = 256;

// The size of the CUDA grid in blocks and blocks in threads

const int nSingleBlocks = 1;
const int nMultiBlocks = 4;
const int nBlocks = nSingleBlocks;
const int nThreads = 256;

// The negation kernel with a single block
__global__
void negate(float *devArray)
{
  // HANDSON 1.7 Get the index of this block in the grid
  int idx = threadIdx.x;
  // HANDSON 1.8 Negate the value at that index
  devArray[idx] = -1. * devArray[idx];
}

__global__
void negateMultiBlock(float *devArray)
{
  // HANDSON 1.9 Calculate the index of this block in the grid
  int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  // HANDSON 1.10 Negate the value at that index
  devArray[idx] = -1. * devArray[idx];
}

// Host main function

int main( )
{
  const int sizeChar = size * sizeof(float);

  // HANDSON 1.1 Allocate the memory for the arrays on the host
  float *hostArray = (float *) malloc(sizeChar);
  float *hostOutput = (float *) malloc(sizeChar);

  // Alocate the memory for the array on the device
  float *devArray;
  myCudaCheck(
	      cudaMalloc(&devArray, sizeChar)
	      );

  // Initialize the input array
  for (int i = 0; i < size; i++) {
    hostArray[i] = i;
    hostOutput[i] = 0;
  }

  // HANDSON 1.2 Copy the data from the host to the device
  myCudaCheck(
	      cudaMemcpy(devArray, hostArray, sizeChar, cudaMemcpyHostToDevice)
	      );

  // HANDSON 1.5 Run the kernel on the GPU
  dim3 blocksPerGrid(nBlocks);
  dim3 threadsPerBlock(nThreads);

  negate<<<blocksPerGrid, threadsPerBlock>>>(devArray);

  // HANDSON 1.6 Synchronise with the device
  myCudaCheck(
	      cudaDeviceSynchronize()
	      );

  // HANDSON 1.3 Copy the results back to the host
  myCudaCheck(
	      cudaMemcpy(hostOutput, devArray, sizeChar, cudaMemcpyDeviceToHost)
	      );

  // Print the result
  std::cout << "Output array:" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << hostOutput[i] << std::endl;
  }

  // HANDSON 1.4 Free the device arrays
  myCudaCheck(
	      cudaFree(devArray)
	      );

  // Free the host arrays
  free(hostArray);
  free(hostOutput);

  return 0;
}