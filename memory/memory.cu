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

// Array size stored in device constant memory
static __constant__ int deviceArraySize_;

// CUDA threads per block
#define nThreads 128;

// Array reversing kernel
__global__
void reverse(float* devA, float* devB)
{
  // temporary array for space for the swap
  __shared__ float tmp[nThreads];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  tmp[nThreads - (threadIdx.x+1)] = devA[idx];
  __syncthreads();

  // Offset to the correct index in the target array
  int blockOffset = deviceArraySize_ - (blockIdx.x + 1)*blockDim.x;
  devB[blockOffset + threadIdx.x] = tmp[threadIdx.x];
}

// Main host function
int main( )
{
  // Array size stored on the host
  const int hostArraySize = 65536;
  // size of the array in char
  size_t sizeChar = hostArraySize * sizeof(float);

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
  for (int i = 0; i < hostArraySize; i++) {
    hostIn[i] = i;
    hostOut[i] = 0;
  }

  // Copy the size of the array from the host to the device.
  myCudaCheck(
	      cudaMemcpyToSymbol( deviceArraySize_, &hostArraySize, sizeof(int))
	      );

  // Copy the input array from the host to the device
  myCudaCheck(
	      cudaMemcpy(devIn, hostIn, sizeChar, cudaMemcpyHostToDevice)
	      );

  // Define the size of the task
  dim3 blocksPerGrid(hostArraySize/nThreads);
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
  for (int i = 0; i < hostArraySize; i++) {
    nCorrect += (hostOut[i] == hostIn[hostArraySize - (i+1)]) ? 1 : 0;
  }
  std::cout << ((nCorrect == hostArraySize) ? "Success! " : "Failure: ");
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