#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

cudaEvent_t start, stop;
const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Store results in these variables.
float effective_bandwidth_gb_s;
float bitonic_sort_step_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;
int kernel_call_count = 0;
double bandwidth;
double numerator;
double denominator;

void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  // NEED TO TIME
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, start, stop);

  CALI_MARK_END(cudaMemcpy_host_to_device);


  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  int j, k;

  // KERNEL BEING CALLED X NUMBER OF TIfMES HERE 
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      // NUMBER OF KERNEL CALLS HERE
      kernel_call_count += 1;
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&bitonic_sort_step_time, start, stop);
  // NEED TO TIME
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, start, stop);

  cudaDeviceSynchronize();

  CALI_MARK_END(cudaMemcpy_device_to_host);

  cudaFree(dev_values);

  numerator = kernel_call_count * 6 * size / 1e9;
  denominator = bitonic_sort_step_time / 1000;

  bandwidth = numerator / denominator;


  printf("Kernel call count %d\n", kernel_call_count);
  printf("Numerator %f\n", numerator);
  printf("Denominator %f\n", denominator);
}


float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

int main(int argc, char *argv[])
{
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;
    
    float *values = (float*) malloc( NUM_VALS * sizeof(float));
    array_fill(values, NUM_VALS);

    cali::ConfigManager mgr;
    mgr.start();

    // INSERT ADAIK CODE HERE
    
    mgr.stop();
    mgr.flush();
}