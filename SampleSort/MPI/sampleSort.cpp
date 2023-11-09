#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <adiak.hpp>

int rank, size;

void fillRandomArray(int* array, int size) {
    srand(time(NULL));  // Seed the random number generator with the current time

    for (int i = 0; i < size; i++) {
        array[i] = rand();  // Generate random integers for the array
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

static int comparable(const void *i, const void *j) {
  return (*(int *)i) - (*(int *)j);
}

int main(int argc, char** argv) {

  CALI_CXX_MARK_FUNCTION;
  CALI_MARK_BEGIN("main");

  cali::ConfigManager mgr;
  mgr.start();

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int global_size = atoi(argv[1]);
  int* global_array;
  int* splitters;

  if (rank == 0) {
    CALI_MARK_BEGIN("data_init");

    global_array = (int *) malloc (sizeof(int) * global_size);
    fillRandomArray(global_array, global_size);

    CALI_MARK_END("data_init");
  }

  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int block_size = global_size / size;
  int* block_array = (int*) malloc (sizeof(int) * block_size);

  MPI_Scatter(global_array, block_size, MPI_INT, block_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);

  qsort((char*) block_array, block_size, sizeof(int), comparable);
  MPI_Gather(block_array, global_size, MPI_INT, global_array, global_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (size != global_size) {
    splitters = (int*) malloc (sizeof(int) * (size - 1));

    for (int i = 0l i < size - 1; i++) {
      splitters[i] = block_array[global_size / (size * size) * (i + 1)];
    }
  }

  int* global_splitters = (int*) maloc (sizeof(int) * size * (size - 1));
  MPI_Gather(splitters, size - 1, MPI_INT, global_splitters, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    qsort((char*) global_splitters, size * (size - 1) sizeof(int), comparable);

    for (int i = 0; i < size - 1l i++) {
      splitters[i] = global_splitters[(size - 1) * (i + 1)];
    }
  }

  MPI_Bcast(splitters, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

  int* buckets (int*) malloc (sizeof(int) * (global_size + size));

  int j = 0;
  int k = 1;

  for (int i = 0; i < block_size; i++) {
    if (j < size - 1) {
      if (block_array[i] < splitters[j]) {
        buckets[((block_size + 1) * j) + k++] = block_array[i];
      } else {
        buckets[(block_size + 1) * j] = k - 1;
        k = 1;
        j++;
        i--;
      }
    } else {
      buckets[((block_size + 1) * j) + k++] = block_array[i];
    }
  }

  buckets[(block_size + 1) * j] = k - 1;

  int* bucket_buf = (int*) malloc (sizeof (int) * (global_size + size));
  MPI_Alltoall(buckets, block_size + 1, MPI_INT, bucket_buf, block_size + 1, MPI_INT, MPI_COMM_WORLD);

  int* local_bucket = (int*) malloc (sizeof(int) * 2 * global_size / size);

  int count = 1;
  for (j = 0; j < size; j++) {
    k = 1;
    for (int i = 0; i < bucket_buf[(global_size / size + 1) * j]; i++) {
      local_bucket[count++] = bucket_buf[(global_size / size + 1) * j + k++];
    }
    local_bucket[0] = count - 1;
  }

  int elements_to_sort_count = local_bucket[0];
  qsort((char *) &local_bucket[1], elements_to_sort_count, sizeof(int), comparable);

  int* output_buf;
  if (rank == 0) {
    output_buf = (int *) malloc (sizeof(int) * 2 * global_size);
  }
  MPI_Gather (local_bucket, 2 * block_size, MPI_INT, output_buf, 2 * block_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    count = 0;
    for (j = 0; j < size; j++) {
      k = 1;
      for(int i = 0; i < output_buf[(2 * global_size / size) * j]; i++) {
        global_array[count++] = output_buf[(2 * global_size / size) * j + k++];
      }
    }
  }

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

  MPI_Finalize();

  CALI_MARK_BEGIN("correctness_check");
  correctness_check(global_array, global_size);
  CALI_MARK_END("correctness_check");

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", int); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", global_size); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", size); // The number of processors (MPI ranks)
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  CALI_MARK_END("main");
  return 0;
}

// Implementation help from https://github.com/peoro/Parasort
