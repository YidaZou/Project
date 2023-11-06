#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

int num_threads;
int num_blocks;
int inputSize;
float *values

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

void sampleSort(float *values){
    

}

void correctness_check(){
    //check if sorted
    int i;
    for(i = 0; i < inputSize - 1; i++){
        if(values[i] > values[i+1]){
            printf("Error: Array not sorted\n");
            return;
        }
    }
    printf("Array sorted correctly\n");
    return;
}

void data_init(){
    values = (float*) malloc(inputSize * sizeof(float));
    array_fill(values, inputSize);
}

void main(int argc, char *argv[]){
    num_threads = atoi(argv[1]);
    inputSize = atoi(argv[2]);
    num_blocks = inputSize/num_threads;

    data_init();

    sampleSort(values);

    correctness_check();

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPIwithCUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 6); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    return;
}

