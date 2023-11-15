#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* main_region = "main_region";
const char* data_init_region = "data_init_region";
const char* comm_region = "comm_region";
const char* comm_small_region = "comm_small_region";
const char* comm_large_region = "comm_large_region";
const char* correctness_check_region = "correctness_check_region";
const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

cudaEvent_t main_time;
cudaEvent_t bitonic_sort_step_start_time;
cudaEvent_t bitonic_sort_step_end_time;
cudaEvent_t host_to_device_start_time;
cudaEvent_t host_to_device_end_time;
cudaEvent_t device_to_host_start_time;
cudaEvent_t device_to_host_end_time;

enum sort_type{
  SORTED,
  REVERSE_SORTED,
  PERTURBED,
  RANDOM
};
