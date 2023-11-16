#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0      

int numtasks, taskid, numworkers, source, dest, mtype;
double whole_computation_time, master_initialization_time = 0;

float *curr_array;
float *global_array;
int array_size;

MPI_Status status;

// Define Caliper region names
const char *main_region = "main";
const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_MPI_Barrier = "comm_MPI_Barrier";
const char *comm_large = "comm_large";
const char *comm_large_MPI_Gather = "comm_large_MPI_Gather";
const char *comm_large_MPI_Scatter = "comm_large_MPI_Scatter";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *correctness_check = "correctness_check";

float random_float() {
    return (float)rand() / (float)RAND_MAX;
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

void array_fill(float *values, float NUM_VALS) {
  if(TYPE == "Random"){
    array_fill_random(values, NUM_VALS);
  }
  else if(TYPE == "Sorted"){
    array_fill_sorted(values, NUM_VALS);
  }
  else if(TYPE == "ReverseSorted"){
    array_fill_reverseSorted(values, NUM_VALS);
  }
  else if(TYPE == "1%perturbed"){
    array_fill_1perturbed(values, NUM_VALS);
  }
  else{
    printf("Error: Invalid input type\n");
    return;
  }
}

bool check_sorted(const float *arr, int length) {
    for (int i = 1; i < length; ++i) {
        if (arr[i - 1] > arr[i]) {
            std::cout << arr[i - 1] << ' ' << arr[i] << ' ';
            return false;
        }
    }
    return true;
}

int compare_floats(const void *a, const void *b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

void bitonic_high(int stage_bit) {
    int index;
    float partner_max_value;

    float *receive_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    int receive_count;

    // Receive the maximum value from the partner process
    MPI_Recv(&partner_max_value, 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int send_count = 0;
    float *send_buffer = (float *)malloc((array_size + 1) * sizeof(float));

    // Send the first element of the current array to the partner process
    MPI_Send(&curr_array[0], 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    // Populate the send buffer with values less than the partner's max_value
    for (index = 0; index < array_size; index++) {
        if (curr_array[index] < partner_max_value) {
            send_buffer[send_count + 1] = curr_array[index];
            send_count++;
        } else {
            break;
        }
    }

    // Exchange data with the partner process
    MPI_Recv(receive_buffer, array_size, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    receive_count = (int)receive_buffer[0];
    send_buffer[0] = (float)send_count;

    // Send the send_buffer to the partner process
    MPI_Send(send_buffer, send_count + 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    // Merge received values into the current array
    for (index = 1; index <= receive_count; index++) {
        if (receive_buffer[index] > curr_array[0]) {
            curr_array[0] = receive_buffer[index];
        } else {
            break;
        }
    }

    // Sort the updated current array
    qsort(curr_array, array_size, sizeof(float), compare_floats);

    free(send_buffer);
    free(receive_buffer);
}


void bitonic_low(int stage_bit) {
    int i;
    float partner_min_value;

    // Allocate memory for send and receive buffers
    float *send_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    int send_count = 0;
    MPI_Send(&curr_array[array_size - 1], 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    int receive_count;
    float *receive_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    MPI_Recv(&partner_min_value, 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Populate send buffer with values greater than partner's min_value
    for (i = array_size - 1; i >= 0; i--) {
        if (curr_array[i] > partner_min_value) {
            send_buffer[send_count + 1] = curr_array[i];
            send_count++;
        } else {
            break;
        }
    }

    send_buffer[0] = (float)send_count;
    MPI_Send(send_buffer, send_count + 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);
    MPI_Recv(receive_buffer, array_size, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Merge received values into the current array
    receive_count = (int)receive_buffer[0];
    for (i = 1; i <= receive_count; i++) {
        if (curr_array[array_size - 1] < receive_buffer[i]) {
            curr_array[array_size - 1] = receive_buffer[i];
        } else {
            break;
        }
    }

    // Sort the updated current array
    qsort(curr_array, array_size, sizeof(float), compare_floats);
    free(send_buffer);
    free(receive_buffer);
}


int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    CALI_MARK_BEGIN(main_region);

    int numVals;

    if (argc == 2) {
        numVals = atoi(argv[1]);
    } else {
        fprintf(stderr, "\nUsage: %s <number_of_values>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize MPI and determine rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    float *values = (float *)malloc(numVals * sizeof(float));

    if (numtasks < 2) {
        fprintf(stderr, "Need at least two MPI tasks to start. Quitting...\n");
        MPI_Finalize();
        free(values);
        return EXIT_FAILURE;
    }

    array_size = numVals / numtasks;
    curr_array = (float *)malloc(array_size * sizeof(float));

    // Allocate memory for the global array in the master process
    if (taskid == MASTER) {
        global_array = (float *)malloc(numVals * sizeof(float));
    }

    CALI_MARK_BEGIN(data_init);
    array_fill(curr_array, array_size);
    CALI_MARK_END(data_init);

    // MPI barrier
    MPI_Barrier(MPI_COMM_WORLD);
    int proc_step = (int)(log2(numtasks));

    // Local sort in worker processes
    qsort(curr_array, array_size, sizeof(float), compare_floats);

    // Iterate over stages, processes, and call bitonic_low or bitonic_high
    for (int i = 0; i < proc_step; i++) {
        for (int j = i; j >= 0; j--) {
            if (((taskid >> (i + 1)) % 2 == 0 && (taskid >> j) % 2 == 0) ||
                ((taskid >> (i + 1)) % 2 != 0 && (taskid >> j) % 2 != 0)) {
                bitonic_low(j);
            } else {
                bitonic_high(j);
            }
        }
    }

    // MPI GATHER for collecting local arrays into the global array
    MPI_Gather(curr_array, array_size, MPI_FLOAT, global_array, array_size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    CALI_MARK_BEGIN(correctness_check);

    // Check the correctness of the global array sorting in the master process
    if (taskid == MASTER) {
        int is_correct = check_sorted(global_array, numVals);
        if (is_correct) {
            printf("The global array is correctly sorted.\n");
        } else {
            printf("Error: The global array is not correctly sorted.\n");
        }
    }
    CALI_MARK_END(correctness_check);

    CALI_MARK_END(main_region);

    free(curr_array);
    if (taskid == MASTER) {
        free(global_array);
    }
    
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Bitonic"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", numVals); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("group_num", "6"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

		
    MPI_Finalize();

    return 0;
}
