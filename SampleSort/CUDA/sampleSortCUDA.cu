#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
std::string TYPE;

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
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
  for(int i = 0; i < NUM_VALS-1; i++){
    if(outValues[i] > outValues[i+1]){
      std::cout << outValues[i] << ' '<< outValues[i+1] << ' ';
      printf("Error: Not sorted\n");
      return;
    }
  }
  printf("Success: Sorted\n");
}

void dataInit(float *values, float NUM_VALS){
  if(TYPE == "Random"){
    array_fill_random(values, NUM_VALS);
  }
  else if(TYPE == "Sorted"){
    array_fill_sorted(values, NUM_VALS);
  }
  else if(TYPE == "ReverseSorted"){
    array_fill_reverseSorted(values, NUM_VALS);
  }
  else if(TYPE == "1perturbed"){
    array_fill_1perturbed(values, NUM_VALS);
  }
  else{
    printf("Error: Invalid input type\n");
    return;
  }
}

int compare(const void *a, const void *b){ //for qsort
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

__global__ void sortSelectSamples(float* dev_values, int NUM_VALS, int sampleSize, float* all_samples, int numSamples) {
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  int start = threadID * sampleSize;

  if(start < NUM_VALS) {
    //sort locally
    for(int i=0; i<sampleSize-1; i++) {
      for(int j=start; j<start+sampleSize-i-1; j++) {
        if(dev_values[j] > dev_values[j+1]) {
          float temp = dev_values[j];
          dev_values[j] = dev_values[j+1];
          dev_values[j+1] = temp;
        }
      }
    }
    //select samples
    for(int i = 0; i < numSamples; i++) {
      all_samples[i+threadID*numSamples] = dev_values[start+i*sampleSize/numSamples];
    }
  }
}

__global__ void findDisplacements(float* dev_values, int NUM_VALS, int BLOCKS, int sampleSize, float* pivots, int* numIncomingValues, int* displacements) {
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  int start = threadID * sampleSize;

  if (start < NUM_VALS) {
    //Get local values in sample
    float* local_values = new float[sampleSize];
    for(int i=start; i<start+sampleSize; i++) {
      local_values[i-start] = dev_values[i];
    }

    int *localCounts = new int[BLOCKS];
    int *localDisplacements = new int[BLOCKS];

    for(int i=0; i<BLOCKS; i++){
      localCounts[i] = 0;
    }

    //Determine placement based on pivots
    for(int i=0; i<sampleSize; i++) {
      bool placed = false;
      for(int k=0; k<BLOCKS-1; k++) {
        if(local_values[i] < pivots[k]) {
          localCounts[k]++;
          placed = true;
          break;
        }
      }
      if(!placed){
        localCounts[BLOCKS-1]++;
      }
    }
      
    //Calculate displacements
    localDisplacements[0] = 0;
    for(int i=1; i<BLOCKS; i++){
      int sum = 0;
      for (int k=i-1; k>=0; k--){
        sum += localCounts[k];
      }
      localDisplacements[i] = sum;
    }

    //Move local to global
    for(int i=0; i < BLOCKS; i++){
      numIncomingValues[i+threadID*BLOCKS] = localCounts[i];
      displacements[i+threadID*BLOCKS] = localDisplacements[i];
    }
  }
}

void selectPivots(float* all_samples, float* pivots, int samples_size, int numSamples) {
  for(int i=numSamples; i < samples_size; i += numSamples) {
    pivots[(i/numSamples)-1] = all_samples[i];
  }
}

void computeFinalCounts(int* incoming_values, int* final_counts, int BLOCKS) {
  for(int i=0; i<BLOCKS; i++) {
    int sum = 0;
    for (int k=i; k<BLOCKS*BLOCKS; k+=BLOCKS){
      sum += incoming_values[k];
    }
    final_counts[i] = sum;
  }
}

__global__ void sendDisplacedValues(float* final_unsorted_values, float* dev_values, int NUM_VALS, int BLOCKS, int sampleSize, int *numIncomingValues, int *displacements, int *final_values){
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  int start = threadID * sampleSize;

  if(start < NUM_VALS) {
    //Get local values in sample
    float* local_values = new float[sampleSize];
    for(int i=start; i<start+sampleSize; i++) {
      local_values[i-start] = dev_values[i];
    }

    //place values in global
    for(int i=0; i<BLOCKS; i++){
      for(int k = displacements[i+threadID*BLOCKS]; k<displacements[i+threadID*BLOCKS]+numIncomingValues[i+threadID*BLOCKS]; k++){ 
        int offset = k - displacements[i+threadID*BLOCKS];
        for(int j=0; j<threadID; j++){
          offset += numIncomingValues[j*BLOCKS+i];
        }       
        if(i>0) {
          for (int n=0; n<i; n++){
            offset += final_values[n];
          }
        }            
        final_unsorted_values[offset] = local_values[k];
      }
    }
  }
}

__global__ void finalSort(float *final_sorted_values, float* final_unsorted_values, int NUM_VALS, int sampleSize, int *final_values) {
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  int start = threadID * sampleSize;

  if (start < NUM_VALS) {
    float *final_local_values = new float[final_values[threadID]];

    int idx = 0;
    for(int i=0; i<threadID; i++){
        idx += final_values[i];
    }

    //Get local values in sample
    for(int i=idx; i<idx+final_values[threadID]; i++) {
        final_local_values[i-idx] = final_unsorted_values[i];
    }

    //Sort values locally
    for (int i=0; i<final_values[threadID]-1; i++) {
      for (int j=0; j<final_values[threadID]-i-1; j++) {
        if (final_local_values[j] > final_local_values[j+1]) {
          float temp = final_local_values[j];
          final_local_values[j] = final_local_values[j + 1];
          final_local_values[j + 1] = temp;
        }
      }
    }

    //Place sorted values in global
    for(int i = 0; i < final_values[threadID]; i++) {
        final_sorted_values[i + idx] = final_local_values[i];
    }
  }
}

void sample_sort(float* values) {
  int sampleSize = NUM_VALS / BLOCKS;
  int numSamples = BLOCKS > sampleSize ? sampleSize / 2 : BLOCKS;
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**)&dev_values, size);

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  float *all_samples;
  size_t allBlocksSize = BLOCKS * numSamples * sizeof(float);
  cudaMalloc((void**)&all_samples, allBlocksSize);

  dim3 blocks(BLOCKS,1); 
  dim3 threads(THREADS,1);
  
  //Sort and select samples
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  sortSelectSamples<<<blocks, threads>>>(dev_values, NUM_VALS, sampleSize, all_samples, numSamples);
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  cudaDeviceSynchronize();

  //Collect all samples from all blocks
  float *final_samples = (float*)malloc(allBlocksSize);
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_small);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(final_samples, all_samples, allBlocksSize, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_small);
  CALI_MARK_END(comm);

  //Select pivots
  size_t pivotSize = (BLOCKS-1) * sizeof(float);
  float *pivots = (float*)malloc(pivotSize);
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_small);
  qsort(final_samples, BLOCKS*numSamples, sizeof(float), compare);    
  selectPivots(final_samples, pivots, BLOCKS*numSamples, numSamples);
  CALI_MARK_END(comp_small);
  CALI_MARK_END(comp);

  //Send pivots
  float *final_pivots;
  cudaMalloc((void**)&final_pivots, pivotSize);
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_small);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(final_pivots, pivots, pivotSize, cudaMemcpyHostToDevice);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_small);
  CALI_MARK_END(comm);

  //Find displacements
  size_t numBlocks2Size = BLOCKS * BLOCKS * sizeof(int);
  int *numIncomingValues;
  cudaMalloc((void**)&numIncomingValues, numBlocks2Size);
  int *displacements;
  cudaMalloc((void**)&displacements, numBlocks2Size);
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  findDisplacements<<<blocks, threads>>>(dev_values, NUM_VALS, BLOCKS, sampleSize, final_pivots, numIncomingValues, displacements);
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  cudaDeviceSynchronize();

  //Communicate number of incoming values in each block
  int* incoming_values = (int*)malloc(numBlocks2Size);
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_small);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(incoming_values, numIncomingValues, numBlocks2Size, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_small);
  CALI_MARK_END(comm);

  //Calculate final values
  size_t finalBlockSize = BLOCKS * sizeof(int);
  int* final_values;
  cudaMalloc((void**)&final_values, finalBlockSize);
  int* final_counts = (int*)malloc(finalBlockSize);
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_small);
  computeFinalCounts(incoming_values, final_counts, BLOCKS);
  CALI_MARK_END(comp_small);
  CALI_MARK_END(comp);

  //Communicate final values
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_small);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(final_values, final_counts, finalBlockSize, cudaMemcpyHostToDevice);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_small);
  CALI_MARK_END(comm);

  //Send displaced values
  size_t finalSize = NUM_VALS * sizeof(float);
  float *final_unsorted_values;
  cudaMalloc((void**)&final_unsorted_values, finalSize);
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  sendDisplacedValues<<<blocks, threads>>>(final_unsorted_values, dev_values, NUM_VALS, BLOCKS, sampleSize, numIncomingValues, displacements, final_values);
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  cudaDeviceSynchronize();

  //Finally, sort each partition
  float *final_sorted_values;
  cudaMalloc((void**)&final_sorted_values, finalSize);
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
  finalSort<<<blocks, threads>>>(final_sorted_values, final_unsorted_values, NUM_VALS, sampleSize, final_values);
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  cudaDeviceSynchronize();

  //Copy sorted array back to values
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);
  cudaMemcpy(values, final_sorted_values, finalSize, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  //Free data
  cudaFree(dev_values);
  cudaFree(final_values);
  cudaFree(displacements);
  cudaFree(numIncomingValues);
  cudaFree(final_pivots);
  cudaFree(all_samples);
  cudaFree(final_sorted_values);
  cudaFree(final_unsorted_values);
}

int main(int argc, char *argv[]) {
  CALI_CXX_MARK_FUNCTION;
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  TYPE = argv[3];
  BLOCKS = NUM_VALS / THREADS;

  float *values = (float*)malloc(NUM_VALS * sizeof(float));

  CALI_MARK_BEGIN(data_init);
  dataInit(values, NUM_VALS);
  CALI_MARK_END(data_init);

  sample_sort(values);

  CALI_MARK_BEGIN(correctness_check);
  correctnessCheck(values);
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
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", TYPE); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "AI+Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
}

