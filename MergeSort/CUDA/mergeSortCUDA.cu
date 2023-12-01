#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <adiak.hpp>
#include <iostream>

int THREADS;
int BLOCKS;
int NUM_VALS;
std::string TYPE;

//////////////////////*********************************//////////////////////////

//////////////////////******** MERGE FUNCTIONS ********//////////////////////////

__device__ inline void Merge(int *values, int *results, int l, int r, int u) {
  int i, j, k;
  i = l;
  j = r;
  k = l;
  while (i < r && j < u) {
    if (values[i] <= values[j]) {
      results[k] = values[i];
      i++;
    } else {
      results[k] = values[j];
      j++;
    }
    k++;
  }

  while (i < r) {
    results[k] = values[i];
    i++;
    k++;
  }

  while (j < u) {
    results[k] = values[j];
    j++;
    k++;
  }
  for (k = l; k < u; k++) {
    values[k] = results[k];
  }
}

__global__ static void MergeSort(int *values, int *results, int size) {
  extern __shared__ int shared[];

  const unsigned int tid = threadIdx.x;
  int k, u, i;

  // Copy input to shared mem.
  shared[tid] = values[tid];

  __syncthreads();

  k = 1;
  while (k < size) {
    i = 1;
    while (i + k <= size) {
      u = i + k * 2;
      if (u > size) {
        u = size + 1;
      }
      Merge(shared, results, i, i + k, u);
      i = i + k * 2;
    }
    k = k * 2;
    __syncthreads();
  }

  values[tid] = shared[tid];
}

//////////////////////******** END OF MERGE FUNCTIONS********//////////////////////////




//////////////////////****************************************//////////////////////////

// random int generator
int random_int() { return (int)rand() / (int)RAND_MAX; }

// fill an array of specified length with random ints using random int generator
void array_fill_random(int *arr, int length) {
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_int();
  }
}

// fill an array of specified length with ints in perfect order
void array_fill_sorted(int *arr, int length) {
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = i;
  }
}

// fill an array of specified length with ints in perfect reverse order
void array_fill_reverseSorted(int *arr, int length) {
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = length - 1 - i;
  }
}

// fill an array of specified length with ints in nearly perfect order (1% of
// values are out of order)
void array_fill_1perturbed(int *arr, int length) {
  srand(time(NULL));
  int i;
  int perturb = std::max(length / 100, 1);  // Ensure perturb is at least 1
  for (i = 0; i < length; ++i) {
    if (i % perturb == 0)
      arr[i] = random_int();
    else
      arr[i] = i;
  }
}

// handle command-line args for specifying data init type
void dataInit(int *values, int NUM_VALS) {
  if (TYPE == "Random") {
    array_fill_random(values, NUM_VALS);
  } else if (TYPE == "Sorted") {
    array_fill_sorted(values, NUM_VALS);
  } else if (TYPE == "ReverseSorted") {
    array_fill_reverseSorted(values, NUM_VALS);
  } else if (TYPE == "1perturbed") {
    array_fill_1perturbed(values, NUM_VALS);
  } else {
    printf("Error: Invalid input type\n");
    return;
  }
}

// Check if correctly sorted
void correctnessCheck(int *outValues) {
  for (int i = 0; i < NUM_VALS - 1; i++) {
    if (outValues[i] > outValues[i + 1]) {
      printf("\nError: Array not sorted\n");
      return;
    }
  }
  printf("\nArray sorted correctly\n");
}

int main(int argc, char *argv[]) {
  CALI_MARK_BEGIN("main");

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  TYPE = argv[3];
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n\n", BLOCKS);

  // cali config manager
  cali::ConfigManager mgr;
  mgr.start();

  int values[NUM_VALS];
  int *dvalues, *results;
  
  CALI_MARK_BEGIN("data_init");
  dataInit(values, NUM_VALS);
  CALI_MARK_END("data_init");

  // Check for CUDA errors
  cudaError_t cudaStatus;

  cudaStatus = cudaMalloc((void **)&dvalues, sizeof(int) * NUM_VALS);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc (dvalues) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }
  cudaStatus = cudaMalloc((void **)&results, sizeof(int) * NUM_VALS);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc (results) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }


  /////// CUDA communication region 1: host to device //////////

  printf("\nCommunication section 1 start: host to device\n");
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaStatus = cudaMemcpy(dvalues, values, sizeof(int) * NUM_VALS, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy (host to device: dvalues) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }
  cudaStatus = cudaMemcpy(results, values, sizeof(int) * NUM_VALS, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy (host to device: results) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");

  /////// End of CUDA communication region 1: host to device //////////




  /////// CUDA computation region //////////

  printf("\nComputation section start\n");
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  MergeSort<<<BLOCKS, THREADS, sizeof(int) * NUM_VALS * 2>>>(dvalues, results, NUM_VALS);
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");
  /////// End of CUDA computation region //////////



  cudaFree(dvalues);



  /////// CUDA communication region 2: device to host //////////
  printf("\nCommunication section 2 start: device to host\n");
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaMemcpy(values, results, sizeof(int) * NUM_VALS, cudaMemcpyDeviceToHost);
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
  /////// End of CUDA communication region 2: device to host //////////



  cudaFree(results);


  CALI_MARK_BEGIN("correctness_check");
  correctnessCheck(values);
  CALI_MARK_END("correctness_check");



  adiak::init(NULL);
  adiak::launchdate();   // launch date of the job
  adiak::libraries();    // Libraries used
  adiak::cmdline();      // Command line used to launch the job
  adiak::clustername();  // Name of the cluster
  adiak::value("Algorithm", "MergeSort");  // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA");  // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int");  // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype",sizeof(int));  // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", NUM_VALS);  // The number of elements in input dataset (1000)
  adiak::value("InputType", TYPE);  // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  // adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
  adiak::value("num_threads", THREADS);  // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS);    // The number of CUDA blocks
  adiak::value("group_num", 6);  // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online");  // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  mgr.stop();
  mgr.flush();

  CALI_MARK_END("main");

  return 0;
}
