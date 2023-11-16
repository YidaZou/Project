#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

#include <adiak.hpp>



//////////////////////*********************************//////////////////////////

//////////////////////******** MERGE FUNCTIONS ********//////////////////////////


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


//////////////////////******** END OF MERGE FUNCTIONS ********//////////////////////////

//////////////////////****************************************//////////////////////////



// random int generator
int random_int()
{
  return (int)rand()/(int)RAND_MAX;
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
    arr[i] = length-1 - i;
  }
}

// fill an array of specified length with ints in nearly perfect order (1% of values are out of order)
void array_fill_1perturbed(int *arr, int length)
{
  srand(time(NULL));
  int i;
  int perturb = length/100;
  for (i = 0; i < length; ++i) {
    if(i % perturb == 0)
      arr[i] = random_int();
    else
      arr[i] = i;
  }
}

// handle command-line args for specifying data init type
void dataInit(int *values, std::string inputType, int inputSize)
{
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

// Function to print an array
void printArray(int A[], int size) 
{
  int i;
  for (i = 0; i < size; i++) printf("%d ", A[i]);
  printf("\n");
}

// Check if correctly sorted
void correctness_check(int arr[], int size) 
{
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

  int rank, size, input_size;
  std::string input_type;

  size = atoi(argv[1]);
  input_size = atoi(argv[2]);
  input_type = argv[3];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int* global_array;

  // Fill the array
  CALI_MARK_BEGIN("data_init");
  global_array = (int *) malloc (sizeof(int) * input_size);
  dataInit(global_array, input_type, input_size);
  CALI_MARK_END("data_init");

  // Perform local merge sort
  int local_size = input_size / size;
  int* local_arr = (int*)malloc(sizeof(int) * local_size);

  // MPI Communication
  CALI_MARK_BEGIN("comm");

  CALI_MARK_BEGIN("MPI_Barrier");
  MPI_Barrier(MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Barrier");

  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("MPI_Scatter");
  MPI_Scatter(&global_array[0], local_size, MPI_INT, &local_arr[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Scatter");
  CALI_MARK_END("comm_large");

  CALI_MARK_END("comm");


  // Merge Computation Region
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  mergeSort(local_arr, 0, local_size - 1);
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");


  // MPI communication region
  CALI_MARK_BEGIN("comm");

  CALI_MARK_BEGIN("MPI_Barrier");
  MPI_Barrier(MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Barrier");

  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("MPI_Gather");
  MPI_Gather(&local_arr[0], local_size, MPI_INT, &global_array[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
  CALI_MARK_END("MPI_Gather");
  CALI_MARK_END("comm_large");

  CALI_MARK_END("comm");


  // Perform final merge on the root process
    if (rank == 0)
    {
        mergeSort(global_array, 0, input_size - 1);

        // Correctness check for the merged array
        CALI_MARK_BEGIN("correctness_check");
        correctness_check(global_array, input_size);
        CALI_MARK_END("correctness_check");
        
    }

    free(global_array);
    free(local_arr);


  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

  MPI_Finalize();

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", input_size); // The number of elements in input dataset (1000)
  adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", size); // The number of processors (MPI ranks)
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  CALI_MARK_END("main");

  return 0;
}
