#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string>

int num_threads;
int num_blocks;
int inputSize;
std::string inputType;

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* cudaMemcpy_region = "cudaMemcpy";

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_fill_random(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

void array_fill_sorted(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = i;
  }
}

void array_fill_reverseSorted(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = length-1 - i;
  }
}

void array_fill_1perturbed(float *arr, int length)
{
  srand(time(NULL));
  int i;
  int perturb = length/100;
  for (i = 0; i < length; ++i) {
    if(i % perturb == 0)
      arr[i] = random_float();
    else
      arr[i] = i;
  }
}

void correctnessCheck(float *outValues){
    //check if sorted
    for(int i = 0; i < inputSize - 1; i++){
        if(outValues[i] > outValues[i+1]){
            printf("Error: Not sorted\n");
            return;
        }
    }
    printf("Success: Sorted\n");
}

void dataInit(float *values){
  if(inputType == "Random"){
    array_fill_random(values, inputSize);
  }
  else if(inputType == "Sorted"){
    array_fill_sorted(values, inputSize);
  }
  else if(inputType == "ReverseSorted"){
    array_fill_reverseSorted(values, inputSize);
  }
  else if(inputType == "1perturbed"){
    array_fill_1perturbed(values, inputSize);
  }
  else{
    printf("Error: Invalid input type\n");
    return;
  }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        std::cerr << "CUDA error in file '" << __FILE__ \
                  << "' in line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void loadShared(float* d_data, float* d_sortedData, float* d_samples, int numSamples, int size) {
    // Each block processes a chunk of the input data
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    extern __shared__ float sharedData[];
    if (tid < size) {
        sharedData[threadIdx.x] = d_data[tid];
    }
}

__global__ void selectSamples(float* d_data, float* d_sortedData, float* d_samples, int numSamples, int size) {
    // Each block selects numSamples - 1 evenly spaced samples
    if (threadIdx.x == 0) {
        for (int i = 0; i < numSamples - 1; ++i) {
            d_samples[blockIdx.x * (numSamples - 1) + i] = sharedData[i * blockDim.x];
        }
    }

    __syncthreads();

    // The first block selects global samples from its samples
    if (blockIdx.x == 0 && threadIdx.x < numSamples - 1) {
        d_samples[numSamples * gridDim.x + threadIdx.x] = d_samples[threadIdx.x * gridDim.x];
    }
}

__global__ void partitionData(float* d_data, float* d_sortedData, float* d_samples, int numSamples, int size) {
    // All blocks use the selected samples to partition their data
    int left = 0;
    int right = (tid < size) ? blockDim.x - 1 : 0;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (sharedData[mid] <= d_samples[threadIdx.x]) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
}

__global__ void sendShared(float* d_data, float* d_sortedData, float* d_samples, int numSamples, int size) {
    // Each block processes a chunk of the input data
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Store the sorted data into global memory
    if (tid < size) {
        d_sortedData[tid] = sharedData[left];
    }
}

// CUDA kernel to perform sample sort
__global__ void sampleSortKernel(float* d_data, float* d_sortedData, float* d_samples, int numSamples, int size) {
    // Each thread sorts its chunk using insertion sort
    for (int i = 1; i < blockDim.x; ++i) {
        float key = sharedData[i];
        int j = i - 1;
        while (j >= 0 && sharedData[j] > key) {
            sharedData[j + 1] = sharedData[j];
            --j;
        }
        sharedData[j + 1] = key;
    }
}

void sampleSort(){
    // Allocate device memory
    float* d_data;
    float* d_sortedData;
    float* d_samples;
    CUDA_CHECK(cudaMalloc((void**)&d_data, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_sortedData, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_samples, numSamples * num_blocks * sizeof(float)));

    // Copy data from host to device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudaMemcpy_region);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CALI_MARK_END(cudaMemcpy_region);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Set up grid and block dimensions
    dim3 blockDim(num_threads);
    dim3 gridDim(num_blocks);

    // Perform sample sort on GPU
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    sampleSort<<<gridDim, blockDim, num_threads * sizeof(float)>>>(d_data, d_sortedData, d_samples, numSamples, inputSize);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    cudaDeviceSynchronize();

    // Load data into shared memory
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    loadShared<<<gridDim, blockDim, num_threads * sizeof(float)>>>(d_data, d_sortedData, d_samples, numSamples, inputSize);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    cudaDeviceSynchronize();

    // Each block selects evenly spaced samples
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    selectSamples<<<gridDim, blockDim, num_threads * sizeof(float)>>>(d_data, d_sortedData, d_samples, numSamples, inputSize);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    cudaDeviceSynchronize();

    // All blocks use the selected samples to partition their data
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    partitionData<<<gridDim, blockDim, num_threads * sizeof(float)>>>(d_data, d_sortedData, d_samples, numSamples, inputSize);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    cudaDeviceSynchronize();

    // Send the sorted data into global memory
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    sendShared<<<gridDim, blockDim, num_threads * sizeof(float)>>>(d_data, d_sortedData, d_samples, numSamples, inputSize);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    cudaDeviceSynchronize();

    // Copy sorted data from device to host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudaMemcpy_region);
    CUDA_CHECK(cudaMemcpy(h_data, d_sortedData, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CALI_MARK_END(cudaMemcpy_region);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
}

int main(int argc, char *argv[]){
    num_threads = atoi(argv[1]);
    inputSize = atoi(argv[2]);
    inputType = argv[3];
    int numSamples = 10;
    num_blocks = (inputSize + num_threads - 1) / num_threads;

    //cali config manager
    cali::ConfigManager mgr;
    mgr.start();

    float* h_data = new float[inputSize];
    CALI_MARK_BEGIN(data_init);
    dataInit(h_data);
    CALI_MARK_END(data_init);

    sampleSort();

    CALI_MARK_BEGIN(correctness_check);
    correctnessCheck(h_data);
    CALI_MARK_END(correctness_check);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI+Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Clean up
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_sortedData));
    CUDA_CHECK(cudaFree(d_samples));
    mgr.stop();
    mgr.flush();

    return 0;
}
