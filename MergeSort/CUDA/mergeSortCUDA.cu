#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* merge_sort_region = "merge_sort";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";



float random_float()
{
  return (float)rand() / (float)RAND_MAX;
}

void array_print(float *arr, int length)
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ", arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

void merge(float *dev_values, int l, int m, int r)
{
  int i, j, k;
  int n1 = m - l + 1;
  int n2 = r - m;

  float *L, *R;
  cudaMalloc((void**)&L, n1 * sizeof(float));
  cudaMalloc((void**)&R, n2 * sizeof(float));

  cudaMemcpy(L, dev_values + l, n1 * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(R, dev_values + m + 1, n2 * sizeof(float), cudaMemcpyDeviceToDevice);

  i = 0;
  j = 0;
  k = l;
  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      dev_values[k] = L[i];
      i++;
    } else {
      dev_values[k] = R[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    dev_values[k] = L[i];
    i++;
    k++;
  }

  while (j < n2) {
    dev_values[k] = R[j];
    j++;
    k++;
  }

  cudaFree(L);
  cudaFree(R);
}

void mergeSort(float *dev_values, int l, int r)
{
  if (l < r) {
    int m = l + (r - l) / 2;
    mergeSort(dev_values, l, m);
    mergeSort(dev_values, m + 1, r);
    merge(dev_values, l, m, r);
  }
}

void mergeSortCuda(float *dev_values, int l, int r)
{
  if (l < r) {
    int m = l + (r - l) / 2;

    mergeSortCuda(dev_values, l, m);
    mergeSortCuda(dev_values, m + 1, r);

    merge(dev_values, l, m, r);
  }
}

// Check if sorted
void correctness_check(float arr[], int size) {
  int i;
  for (i = 0; i < size - 1; i++) {
    if (arr[i] > arr[i + 1]) {
      printf("\nError: Array not sorted\n");
      return;
    }
  }
  printf("\nArray sorted correctly\n");
  return;
}

int main(int argc, char *argv[])
{
  CALI_MARK_BEGIN("main");

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;


  // Create Caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  // Data initialization
  CALI_MARK_BEGIN("data_init");
  float *values = (float *)malloc(NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void **)&dev_values, size);

  CALI_MARK_END("data_init");


  // MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  CALI_MARK_END(cudaMemcpy_host_to_device);

  dim3 blocks(BLOCKS, 1); /* Number of blocks */
  dim3 threads(THREADS, 1); /* Number of threads */

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  mergeSortCuda(dev_values, 0, NUM_VALS - 1);
  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  CALI_MARK_END(cudaMemcpy_device_to_host);
  
  // Correctness check for the merged array 
  CALI_MARK_BEGIN("correctness_check");
  correctness_check(values, NUM_VALS);
  CALI_MARK_END("correctness_check");


  cudaFree(dev_values);

  float merge_sort_time = 0.0;
  float cudaMemcpy_host_to_device_time = 0.0;
  float cudaMemcpy_device_to_host_time = 0.0;


  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", float); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", 1000); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "AI") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  // Flush Caliper output
  mgr.stop();
  mgr.flush();

  CALI_MARK_END("main");
}