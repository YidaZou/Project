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


// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
__device__ void merge(int *arr, int *temp, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    // Ensure all threads have finished their work before proceeding
    __syncthreads();

    // Copy the remaining elements from the left subarray
    while (i <= mid)
    {
        temp[k++] = arr[i++];
    }

    // Copy the remaining elements from the right subarray
    while (j <= right)
    {
        temp[k++] = arr[j++];
    }

    // Ensure all threads have finished their work before proceeding
    __syncthreads();

    // Copy the merged elements back to the original array
    arr[left + threadIdx.x] = temp[left + threadIdx.x];
}



// l is for left index and r is right index of the
// sub-array of arr to be sorted
__global__ void mergeSort(int *arr, int *temp, int size)
{
    for (int currSize = 1; currSize <= size - 1; currSize = 2 * currSize)
    {
        for (int leftStart = 0; leftStart < size; leftStart += 2 * currSize)
        {
            int mid = min(leftStart + currSize - 1, size - 1);
            int rightEnd = min(leftStart + 2 * currSize - 1, size - 1);
            merge(arr, temp, leftStart, mid, rightEnd);
        }
    }
}


//////////////////////******** END OF MERGE FUNCTIONS ********//////////////////////////

//////////////////////****************************************//////////////////////////


// random int generator
int random_int() 
{ 
    return (int)rand() / (int)RAND_MAX; 
}

// fill an array of specified length with random ints using random int generator
void array_fill_random(int *arr, int length) 
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_int();
  }
}

// fill an array of specified length with ints in perfect order
void array_fill_sorted(int *arr, int length) 
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = i;
  }
}

// fill an array of specified length with ints in perfect reverse order
void array_fill_reverseSorted(int *arr, int length) 
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = length - 1 - i;
  }
}

// fill an array of specified length with ints in nearly perfect order (1% of values are out of order)
void array_fill_1perturbed(int *arr, int length) 
{
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
void dataInit(int *values, int NUM_VALS) 
{
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
void correctnessCheck(int *outValues) 
{
  for (int i = 0; i < NUM_VALS - 1; i++) {
    if (outValues[i] > outValues[i + 1]) {
      std::cout << outValues[i] << ' ' << outValues[i + 1] << ' ';
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

  // cali config manager
  cali::ConfigManager mgr;
  mgr.start();


  int *host_arr = new int[NUM_VALS];
  int *device_arr;
  int *temp;

  CALI_MARK_BEGIN("data_init");
  dataInit(host_arr, NUM_VALS);
  CALI_MARK_END("data_init");

  // Check for CUDA errors
  cudaError_t cudaStatus;
 


  /////// CUDA communication region //////////
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaStatus = cudaMalloc((void **)&device_arr, sizeof(int) * NUM_VALS);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed! Error: %s\n", cudaGetErrorString(cudaStatus));
      return 1;
  }
  cudaStatus = cudaMemcpy(device_arr, host_arr, sizeof(int) * NUM_VALS, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy (host to device) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
      return 1;
  }
  cudaStatus = cudaMalloc((void **)&temp, sizeof(int) * NUM_VALS);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc (temp) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
      return 1;
  }
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
  /////// End of CUDA communication region //////////



  /////// CUDA computation region //////////
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  mergeSort<<<BLOCKS, THREADS>>>(device_arr, temp, NUM_VALS);
  cudaDeviceSynchronize();
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");
  /////// End of CUDA computation region //////////


  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaStatus = cudaMemcpy(host_arr, device_arr, sizeof(int) * NUM_VALS, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy (device to host) failed! Error: %s\n", cudaGetErrorString(cudaStatus));
      return 1;
  }
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");


  CALI_MARK_BEGIN("correctness_check");
  correctnessCheck(host_arr);
  CALI_MARK_END("correctness_check");

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", TYPE); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  //adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  delete[] host_arr;
  cudaFree(device_arr);
  cudaFree(temp);

  mgr.stop();
  mgr.flush();

  CALI_MARK_END("main");

  return 0;
}
