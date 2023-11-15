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
