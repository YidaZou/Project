#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

int num_threads;
int num_blocks;
int inputSize;
int bucketSize;
char* inputType;

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
//const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
//const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

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

void correctness_check(float *outValues){
    //check if sorted
    for(int i = 0; i < inputSize - 1; i++){
        if(outValues[i] > outValues[i+1]){
            printf("Error: Not sorted\n");
            return;
        }
    }
    printf("Success: Sorted\n");
}

void data_init(float *values){
  if(inputType == "Random"){
    array_fill_random(values, inputSize);
  }
  else if(inputType == "Sorted"){
    array_fill_sorted(values, inputSize);
  }
  else if(inputType == "ReverseSorted"){
    array_fill_reverseSorted(values, inputSize);
  }
  else if(inputType == "1%perturbed"){
    array_fill_1perturbed(values, inputSize);
  }
  else{
    printf("Error: Invalid input type\n");
    return;
  }
}

__global__ void sampleSortKernel(float *values, float *outValues, float* inputSize){
  __shared__ float localBucket[bucketSize];
	__shared__ int localCount; //Counter to track bucket index

	int threadId = threadIdx.x; 
  int blockId = blockIdx.x;
	int offset = blockDim.x;
	int bucket, index, phase;
	float temp;
	
	if(threadId == 0)
		localCount = 0;l

	__syncthreads();

  CALI_MARK_BEGIN(comp_small);
	/* Block traverses through the array and buckets the element accordingly */
	while(threadId < inputSize) {
		bucket = values[threadId] * 10;
		if(bucket == blockId) {
			index = atomicAdd(&localCount, 1);
			localBucket[index] = values[threadId]; 
		}
		threadId += offset;		
	}
  CALI_MARK_END(comp_small);

	__syncthreads();
	
  CALI_MARK_BEGIN(comp_large);
	threadId = threadIdx.x;
	//Sorting the bucket using Parallel Bubble Sort
	for(phase = 0; phase < bucketLength; phase ++) {
		if(phase % 2 == 0) {
			while((threadId < bucketLength) && (threadId % 2 == 0)) {
				if(localBucket[threadId] > localBucket[threadId +1]) {
					temp = localBucket[threadId];
					localBucket[threadId] = localBucket[threadId + 1];
					localBucket[threadId + 1] = temp;
				}
				threadId += offset;
			}
		}
		else {
			while((threadId < bucketLength - 1) && (threadId %2 != 0)) {
				if(localBucket[threadId] > localBucket[threadId + 1]) {
					temp = localBucket[threadId];
					localBucket[threadId] = localBucket[threadId + 1];
					localBucket[threadId + 1] = temp;
				}
				threadId += offset;
			}
		}
	}
  CALI_MARK_END(comp_large);
	
  CALI_MARK_BEGIN(comp_small);
	threadId = threadIdx.x;
	while(threadId < bucketLength) {
		outData[(blockIdx.x * bucketLength) + threadId] = localBucket[threadId];
		threadId += offset;
	}
  CALI_MARK_END(comp_small);

}

void sampleSort(float *values, float outValues){
  float *dev_values, *dev_outValues;
  size_t size = inputSize * sizeof(float);
  size_t outSize = bucketSize * num_blocks sizeof(float);

  cudaMalloc((void**) &dev_values, size);
	cudaMalloc((void**) &dev_outValues, outSize);
	cudaMemset(dev_outValues, 0, outSize);

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  CALI_MARK_BEGIN(comp);
	sampleSortKernel<<<num_blocks, num_threads>>>(dev_values, dev_outValues, inputSize);
  CALI_MARK_END(comp);


  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
	cudaMemcpy(outValues, d_output, out_size, cudaMemcpyDeviceToHost);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  cudaFree(dev_values);
  cudaFree(dev_outValues);
}

void main(int argc, char *argv[]){
    num_threads = atoi(argv[1]);
    inputSize = atoi(argv[2]);
    inputType = argv[3];
    num_blocks = inputSize/num_threads;
    bucketSize = inputSize/num_blocks;

    //cali config manager
    cali::ConfigManager mgr;
    mgr.start();

    float *values = (float*) malloc(inputSize * sizeof(float));
    float *outValues = (float*) malloc(inputSize * sizeof(float));

    CALI_MARK_BEGIN(data_init);
    data_init(values);
    CALI_MARK_END(data_init);

    sampleSort(values, outValues);

    CALI_MARK_BEGIN(correctness_check);
    correctness_check(outValues);
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
    adiak::value("implementation_source", "online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    mgr.stop();
    mgr.flush();
}

