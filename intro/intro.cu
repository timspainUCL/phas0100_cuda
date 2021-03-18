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
  int idx = threadIdx.x;
  devArray[idx] = -1. * devArray[idx];
}

__global__
void negateMultiBlock(float *devArray)
{
  int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  devArray[idx] = -1. * devArray[idx];
}

// Host main function

int main( )
{
  const int sizeChar = size * sizeof(float);

  // Allocate the memory for the arrays on the host
  float *hostArray = (float *) malloc(sizeChar);
  float *hostOutput = (float *) malloc(sizeChar);

  // Alocate the memory for the array on the device
  float *devArray;
  myCudaCheck( cudaMalloc(&devArray, sizeChar) );

  // Initialize the input array
  for (int i = 0; i < size; i++) {
    hostArray[i] = i;
    hostOutput[i] = 0;
  }

  // Copy the data from the host to the device
  myCudaCheck( cudaMemcpy(devArray, hostArray, sizeChar, cudaMemcpyHostToDevice) );

  // Run the kernel on the GPU
  dim3 blocksPerGrid(nBlocks);
  dim3 threadsPerBlock(nThreads);

  negate<<<blocksPerGrid, threadsPerBlock>>>(devArray);

  // Synchronise with the device
  myCudaCheck( cudaDeviceSynchronize() );

  // Copy the results back to the host
  myCudaCheck( cudaMemcpy(hostOutput, devArray, sizeChar, cudaMemcpyDeviceToHost) );

  // Print the result
  std::cout << "Output array:" << std::endl;
  for (int i = 0; i < size; i++) {
    std::cout << hostOutput[i] << std::endl;
  }

  // Free the device arrays
  myCudaCheck( cudaFree(devArray) );

  // Free the host arrays
  free(hostArray);
  free(hostOutput);

  return 0;
}