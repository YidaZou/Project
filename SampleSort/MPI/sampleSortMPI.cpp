#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <adiak.hpp>


// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge(int arr[], int l, int m, int r) {
  int i, j, k;
  int n1 = m - l + 1;
  int n2 = r - m;

  // Create temp arrays
  int L[n1], R[n2];

  // Copy data to temp arrays L[] and R[]
  for (i = 0; i < n1; i++) L[i] = arr[l + i];
  for (j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

  // Merge the temp arrays back into arr[l..r
  i = 0;
  j = 0;
  k = l;
  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      arr[k] = L[i];
      i++;
    } else {
      arr[k] = R[j];
      j++;
    }
    k++;
  }

  // Copy the remaining elements of L[],
  // if there are any
  while (i < n1) {
    arr[k] = L[i];
    i++;
    k++;
  }

  // Copy the remaining elements of R[],
  // if there are any
  while (j < n2) {
    arr[k] = R[j];
    j++;
    k++;
  }
}

// l is for left index and r is right index of the
// sub-array of arr to be sorted
void mergeSort(int arr[], int l, int r) {
  if (l < r) {
    int m = l + (r - l) / 2;

    // Sort first and second halves
    mergeSort(arr, l, m);
    mergeSort(arr, m + 1, r);

    merge(arr, l, m, r);
  }
}

void fillRandomArray(int arr[], int size) {
    srand(time(NULL));  // Seed the random number generator with the current time

    for (int i = 0; i < size; i++) {
        arr[i] = rand();  // Generate random integers for the array
    }
}

// Function to print an array
void printArray(int A[], int size) {
  int i;
  for (i = 0; i < size; i++) printf("%d ", A[i]);
  printf("\n");
}

// Check if sorted
void correctness_check(int arr[], int size) {
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



int main(int argc, char** argv) {
  CALI_MARK_BEGIN("main");

  cali::ConfigManager mgr;
  mgr.start();

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Define an array with 1000 elements
  int local_size = 1000;
  int local_arr[local_size];

  // Fill the array with random values
  CALI_MARK_BEGIN("data_init");
  fillRandomArray(local_arr, local_size);
  CALI_MARK_END("data_init");


  // MPI communication regions
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  MPI_Barrier(MPI_COMM_WORLD);

  // int n = local_size;
  // int* arr = NULL;

  // if (rank == 0) {
  //   arr = (int*)malloc(sizeof(int) * size * n);
  // }

  // MPI_Gather(local_arr, n, MPI_INT, arr, n, MPI_INT, 0, MPI_COMM_WORLD);

  // if (rank == 0) {
  //   // Scatter data to all processes
  //   MPI_Scatter(arr, n, MPI_INT, local_arr, n, MPI_INT, 0, MPI_COMM_WORLD);
  // }

  // CALI_MARK_END("comm_large");
  // CALI_MARK_END("comm");

  // // Merge and check correctness
  // if (rank == 0) {

  //   CALI_MARK_BEGIN("comp");
  //   CALI_MARK_BEGIN("comp_large");
  //   mergeSort(local_arr, 0, local_size - 1);
  //   CALI_MARK_END("comp_large");
  //   CALI_MARK_END("comp");

  //   // Correctness check for the merged array
  //   CALI_MARK_BEGIN("correctness_check");
  //   correctness_check(local_arr, local_size);
  //   CALI_MARK_END("correctness_check");

  // }

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

  MPI_Finalize();

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", int); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", 1000); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", size); // The number of processors (MPI ranks)
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "AI") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  CALI_MARK_END("main");
  return 0;
}
