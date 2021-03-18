/*
 * An exercise on the different types of memory available in CUDA
 */

#include <iostream>
#include <cstdlib>

// Error checking macro function
#define myCudaCheck(result) { cudaErrorCheck((result), __FILE__, __LINE__); }
inline void cudaErrorCheck(cudaError_t err, const char* file, int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
    exit(err);
  }
}

// Array size
// HANDSON 2.1 Change the array size to a static __constant__ int
#define ARRAY_SIZE 65536

// CUDA threads per block
#define nThreads 128

// Array reversing kernel
__global__
void reverse(float* devA, float* devB)
{
  // HANDSON 2.3 Create a __shared__ temporary array of length nThreads for the swap
  
  // Get the index in this block
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // HANDSON 2.4 Fill the temporary array

  // HANDSON 2.5 synchronize the threads

  // HANDSON 2.6 Calculate the initial position of this block in the grid
  // int blockOffset
  // Reverse the elements
  // HANDSON 2.7 Fill the output array with the reversed elements from this block
  devB[idx] = devA[ARRAY_SIZE - (idx + 1)];
}

// Main host function
int main( )
{
  // HANDSON 2.2 Replace the host array size by a const int
  //             Here and elsewhere
  // size of the array in char
  size_t sizeChar = ARRAY_SIZE * sizeof(float);

  // Allocate host memory
  float* hostIn = (float*) malloc(sizeChar);
  float* hostOut = (float*) malloc(sizeChar);

  // Allocate device memory
  float* devIn;
  float* devOut;
  myCudaCheck(
	      cudaMalloc(&devIn, sizeChar)
	      );
  myCudaCheck(
	      cudaMalloc(&devOut, sizeChar)
	      );

  // Initialize the arrays
  for (int i = 0; i < ARRAY_SIZE; i++) {
    hostIn[i] = i;
    hostOut[i] = 0;
  }

  // Copy the input array from the host to the device
  myCudaCheck(
	      cudaMemcpy(devIn, hostIn, sizeChar, cudaMemcpyHostToDevice)
	      );

  // Define the size of the task
  dim3 blocksPerGrid(ARRAY_SIZE/nThreads);
  dim3 threadsPerBlock(nThreads);

  reverse<<<blocksPerGrid, threadsPerBlock>>>(devIn, devOut);

  // Wait for all threads to complete
  myCudaCheck(
	      cudaDeviceSynchronize()
	      );

  // Copy the result array back to the host
  myCudaCheck(
	      cudaMemcpy(hostOut, devOut, sizeChar, cudaMemcpyDeviceToHost)
	      );

  // Check and print the result
  int nCorrect = 0;
  for (int i = 0; i < ARRAY_SIZE; i++) {
    nCorrect += (hostOut[i] == hostIn[ARRAY_SIZE - (i+1)]) ? 1 : 0;
  }
  std::cout << ((nCorrect == ARRAY_SIZE) ? "Success! " : "Failure: ");
  std::cout << nCorrect  << " elements were correctly swapped." << std::endl;

  // Free device memory
  myCudaCheck(
	      cudaFree(devIn)
	      );
  myCudaCheck(
	      cudaFree(devOut)
	      );

  // Free host memory
  free(hostIn);
  free(hostOut);

  return 0;
}