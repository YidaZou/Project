#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
std::string TYPE;
int TYPE_ID;

const char* type[4] = {"random", "sorted", "reverse_sorted", "1perturbed"};


cudaEvent_t start, stop;
const char* main_region = "main";
const char* data_init= "data_init";
const char* comm_region = "comm";
const char* comm_large_region = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* cudaMemcpy = "cudaMemcpy";


const char* correctness_check = "correctness_check";


/*
CUDA Regions needed:
- 
*/


// Store results in these variables.
float bitonic_sort_step_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;
int kernel_call_count = 0;
double numerator;
double denominator;

std::string get_type(int type_id){
  if(type_id == 1){
    return "Random";
  }
  else if(type_id == 2){
    return "Sorted";
  }
  else if(type_id == 3){
    return "ReverseSorted";
  }
  else if(type_id == 4){
    return "1perturbed";
  }
  else{
    return "Invalid";
  }
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
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

void dataInit(float *values, float NUM_VALS){
  if(TYPE == "Random"){
    array_fill_random(values, NUM_VALS);
  }
  else if(TYPE == "Sorted"){
    array_fill_sorted(values, NUM_VALS);
  }
  else if(TYPE == "ReverseSorted"){
    array_fill_reverseSorted(values, NUM_VALS);
  }
  else if(TYPE == "1perturbed"){
    array_fill_1perturbed(values, NUM_VALS);
  }
  else{
    printf("Error: Invalid input type\n");
    return;
  }
}

void correctnessCheck(float *outValues){
  //check if sorted
  for(int i = 0; i < NUM_VALS-1; i++){
    if(outValues[i] > outValues[i+1]){
      std::cout << outValues[i] << ' '<< outValues[i+1] << ' ';
      printf("Error: Not sorted\n");
      return;
    }
  }
  printf("Success: Sorted\n");
}
// NEED TO TIME THIS FUNCTION
__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  // NEED TO TIME
  //MEM COPY FROM HOST TO DEVICE



  CALI_MARK_BEGIN(comm_region);
  CALI_MARK_BEGIN(comm_large_region);
  CALI_MARK_BEGIN(cudaMemcpy);

  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  
  CALI_MARK_END(comm_region);
  CALI_MARK_END(cudaMemcpy);
  CALI_MARK_END(comm_large_region);


  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */


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
  cudaDeviceSynchronize();


  // NEED TO TIME
  //MEM COPY FROM DEVICE TO HOST


  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);




  cudaFree(dev_values);
}


int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  TYPE_ID = atoi(argv[3]);
  TYPE = get_type(TYPE_ID);

  BLOCKS = NUM_VALS / THREADS;

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();
  CALI_MARK_BEGIN(main_region);

  float *values = (float*) malloc( NUM_VALS * sizeof(float));

  CALI_MARK_BEGIN(data_init);
  dataInit(values, NUM_VALS);
  CALI_MARK_END(data_init);
  
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);

  bitonic_sort(values); /* Inplace */

  CALI_MARK_END(comp);
  CALI_MARK_END(comp_large);

  CALI_MARK_BEGIN(correctness_check);
  correctnessCheck(values);
  CALI_MARK_END(correctness_check);

  CALI_MARK_END(main_region);


  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "Bitonic"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", type[TYPE_ID - 1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("group_num", "6"); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Lab"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}