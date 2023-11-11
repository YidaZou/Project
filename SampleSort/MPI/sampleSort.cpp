#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <adiak.hpp>

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

void dataInit(float *values, int inputSize){
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
  return (*(float *)i) - (*(float *)j);
}

int main(int argc, char** argv) {

  int rank, size, input_size;
  std::string input_type;

  size = atoi(argv[1]);
  input_size = atoi(argv[2]);
  input_type = argv[3];

  CALI_CXX_MARK_FUNCTION;
  CALI_MARK_BEGIN("main");

  cali::ConfigManager mgr;
  mgr.start();

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  float* global_array;
  float* splitters;

  if (rank == 0) {
    CALI_MARK_BEGIN("data_init");

    global_array = (float *) malloc (sizeof(float) * input_size);
    dataInit(global_array, input_size);

    CALI_MARK_END("data_init");
  }

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_small");

  MPI_Bcast(&input_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_small");

  int block_size = input_size / size;
  float* block_array = (float*) malloc (sizeof(float) * block_size);

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  MPI_Scatter(global_array, block_size, MPI_INT, block_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_large");

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");

  qsort((char*) block_array, block_size, sizeof(float), comparable);

  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  MPI_Gather(block_array, input_size, MPI_INT, global_array, input_size, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_large");

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_small");

  if (size != input_size) {
    splitters = (float*) malloc (sizeof(float) * (size - 1));

    for (int i = 0l i < size - 1; i++) {
      splitters[i] = block_array[input_size / (size * size) * (i + 1)];
    }
  }

  CALI_MARK_END("comp");
  CALI_MARK_END("comp_small");

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_small");

  float* global_splitters = (float*) maloc (sizeof(float) * size * (size - 1));
  MPI_Gather(splitters, size - 1, MPI_INT, global_splitters, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_small");

  if (rank == 0) {

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");

    qsort((char*) global_splitters, size * (size - 1) sizeof(float), comparable);

    for (int i = 0; i < size - 1l i++) {
      splitters[i] = global_splitters[(size - 1) * (i + 1)];
    }

    CALI_MARK_END("comp");
    CALI_MARK_END("comp_small");
  }

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_small");

  MPI_Bcast(splitters, size - 1, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_small");

  float* buckets = (float*) malloc (sizeof(float) * (input_size + size));


  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");

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

  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  float* bucket_buf = (float*) malloc (sizeof (int) * (input_size + size));
  MPI_Alltoall(buckets, block_size + 1, MPI_INT, bucket_buf, block_size + 1, MPI_INT, MPI_COMM_WORLD);

  float* local_bucket = (float*) malloc (sizeof(float) * 2 * input_size / size);

  int count = 1;
  for (j = 0; j < size; j++) {
    k = 1;
    for (int i = 0; i < bucket_buf[(input_size / size + 1) * j]; i++) {
      local_bucket[count++] = bucket_buf[(input_size / size + 1) * j + k++];
    }
    local_bucket[0] = count - 1;
  }

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_large");

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");

  int elements_to_sort_count = local_bucket[0];
  qsort((char *) &local_bucket[1], elements_to_sort_count, sizeof(float), comparable);

  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");

  float* output_buf;
  if (rank == 0) {
    output_buf = (float *) malloc (sizeof(float) * 2 * input_size);
  }
  MPI_Gather (local_bucket, 2 * block_size, MPI_INT, output_buf, 2 * block_size, MPI_INT, 0, MPI_COMM_WORLD);

  CALI_MARK_END("comm");
  CALI_MARK_END("comm_large");

  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");

  if (rank == 0) {
    count = 0;
    for (j = 0; j < size; j++) {
      k = 1;
      for(int i = 0; i < output_buf[(2 * input_size / size) * j]; i++) {
        global_array[count++] = output_buf[(2 * input_size / size) * j + k++];
      }
    }
  }

  CALI_MARK_END("comp");
  CALI_MARK_END("comp_large");

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

  MPI_Finalize();

  CALI_MARK_BEGIN("correctness_check");
  correctness_check(global_array, input_size);
  CALI_MARK_END("correctness_check");

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
  adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
  adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  CALI_MARK_END("main");
  return 0;
}

// Implementation help from https://github.com/peoro/Parasort
