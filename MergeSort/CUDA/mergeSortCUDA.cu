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

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
    srand(time(NULL));
    for (int i = 0; i < length; ++i) {
        arr[i] = random_float();
    }
}

__global__ void mergeSortKernel(float* values, int left, int right)
{
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Recursive calls
        CALI_MARK_BEGIN("comp_large");
        mergeSortKernel(values, left, mid);
        mergeSortKernel(values, mid + 1, right);
        CALI_MARK_END("comp_large");

        // Merge step
        CALI_MARK_BEGIN("comp_small");
        merge(values, left, mid, right);
        CALI_MARK_END("comp_small");
    }
}

void merge(float* values, int left, int mid, int right)
{
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    float* L, * R;
    cudaMalloc((void**)&L, n1 * sizeof(float));
    cudaMalloc((void**)&R, n2 * sizeof(float));

    cudaMemcpy(L, values + left, n1 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(R, values + mid + 1, n2 * sizeof(float), cudaMemcpyDeviceToDevice);

    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            values[k] = L[i];
            i++;
        }
        else {
            values[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        values[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        values[k] = R[j];
        j++;
        k++;
    }

    cudaFree(L);
    cudaFree(R);
}

void mergeSort(float* values, float* outValues, int left, int right)
{
    float *dev_values;
    size_t size = inputSize * sizeof(float);

    cudaMalloc((void**)&dev_values, size);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    mergeSortKernel<<<num_blocks, num_threads>>>(dev_values, left, right);
    cudaDeviceSynchronize();  // Ensure kernel completion
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(outValues, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    cudaFree(dev_values);
}

void data_init(float* values)
{
    array_fill(values, inputSize);
}

void correctness_check(float* outValues)
{
    // Check if sorted
    for (int i = 0; i < inputSize - 1; i++) {
        if (outValues[i] > outValues[i + 1]) {
            printf("Error: Not sorted\n");
            return;
        }
    }
    printf("Success: Sorted\n");
}

int main(int argc, char* argv[])
{
    num_threads = atoi(argv[1]);
    inputSize = atoi(argv[2]);
    num_blocks = inputSize / num_threads;

    // cali config manager
    cali::ConfigManager mgr;
    mgr.start();

    float *values = (float*)malloc(inputSize * sizeof(float));
    float *outValues = (float*)malloc(inputSize * sizeof(float));

    CALI_MARK_BEGIN("data_init");
    data_init(values);
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    mergeSort(values, outValues, 0, inputSize - 1);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("correctness_check");
    correctness_check(outValues);
    CALI_MARK_END("correctness_check");

    adiak::init(NULL);
    adiak::launchdate();  // launch date of the job
    adiak::libraries();   // Libraries used
    adiak::cmdline();     // Command line used to launch the job
    adiak::clustername(); // Name of the cluster
    adiak::value("Algorithm", "MergeSort");          // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPIwithCUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");               // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float));   // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);            // The number of elements in the input dataset (1000)
    adiak::value("InputType", "Random");             // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", num_threads);        // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks);          // The number of CUDA blocks
    adiak::value("group_num", 6);                    // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
    return 0;
}
