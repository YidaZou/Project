#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
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