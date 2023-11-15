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

void array_fill(float *arr, int length) {
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = random_float();
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

int compare_floats(const void *a_ptr, const void *b_ptr) {
    float value_a = *(const float *)a_ptr;
    float value_b = *(const float *)b_ptr;

    if (value_a < value_b) {
        return -1;
    } else if (value_a > value_b) {
        return 1;
    } else {
        return 0;
    }
}

void bitonic_high(int stage_bit) {
    int index;
    float helper_max;

    // Allocate buffers for send and receive
    float *receive_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    int receive_count;

    // Receive helper_max from the partner process
    MPI_Recv(&helper_max, 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Prepare send buffer
    int send_count = 0;
    float *send_buffer = (float *)malloc((array_size + 1) * sizeof(float));

    // Send the first element of curr_array to the partner process
    MPI_Send(&curr_array[0], 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    // Populate send buffer with values less than partner's max_value
    for (index = 0; index < array_size; index++) {
        if (curr_array[index] < helper_max) {
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

    // Merge received values
    for (index = 1; index <= receive_count; index++) {
        if (receive_buffer[index] > curr_array[0]) {
            curr_array[0] = receive_buffer[index];
        } else {
            break;
        }
    }

    // Sort the updated local array
    qsort(curr_array, array_size, sizeof(float), compare_floats);

    // Free dynamically allocated memory
    free(send_buffer);
    free(receive_buffer);
}

void bitonic_low(int stage_bit) {
    int i;
    float helper_min;

    // Allocate buffers for send and receive
    float *send_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    int send_count = 0;
    MPI_Send(&curr_array[array_size - 1], 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);

    int receive_count;
    float *receive_buffer = (float *)malloc((array_size + 1) * sizeof(float));
    MPI_Recv(&helper_min, 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Populate send buffer with values greater than partner's min
    for (i = array_size - 1; i >= 0; i--) {
        if (curr_array[i] > helper_min) {
            send_buffer[send_count + 1] = curr_array[i];
            send_count++;
        } else {
            break;
        }
    }

    send_buffer[0] = (float)send_count;
    MPI_Send(send_buffer, send_count + 1, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD);
    MPI_Recv(receive_buffer, array_size, MPI_FLOAT, taskid ^ (1 << stage_bit), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Merge received values
    receive_count = (int)receive_buffer[0];
    for (i = 1; i <= receive_count; i++) {
        if (curr_array[array_size - 1] < receive_buffer[i]) {
            curr_array[array_size - 1] = receive_buffer[i];
        } else {
            break;
        }
    }

    // Sort the updated local array
    qsort(curr_array, array_size, sizeof(float), compare_floats);
    free(send_buffer);
    free(receive_buffer);
};


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

    // data initialization
    array_size = numVals / numtasks;

    curr_array = (float *)malloc(array_size * sizeof(float));
    if (taskid == MASTER) {
        global_array = (float *)malloc(numVals * sizeof(float));
    }

    CALI_MARK_BEGIN(data_init);
    array_fill(curr_array, array_size);
    CALI_MARK_END(data_init);

    // MP barrier
    MPI_Barrier(MPI_COMM_WORLD);
    int proc_step = (int)(log2(numtasks));

    // local sort in worker
    qsort(curr_array, array_size, sizeof(float), compare_floats);

    // iterate over stages, processes, and call high or low
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

    // MPI GATHER for local to global
    MPI_Gather(curr_array, array_size, MPI_FLOAT, global_array, array_size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

    CALI_MARK_BEGIN(correctness_check);

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
    MPI_Finalize();

    return 0;
}
