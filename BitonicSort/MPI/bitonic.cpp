#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int numtasks, taskid, numworkers, source, dest, mtype;
double whole_computation_time, master_initialization_time = 0;

float *local_array;
float *global_array;
int array_size;

MPI_Status status;

// Define Caliper region names
const char *main_region = "main";
const char *comm = "comm";
const char *comm_MPI_Barrier = "comm_MPI_Barrier";
const char *comm_large = "comm_large";
const char *comm_large_MPI_Gather = "comm_large_MPI_Gather";
const char *comm_large_MPI_Scatter = "comm_large_MPI_Scatter";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *data_init = "data_init";

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
