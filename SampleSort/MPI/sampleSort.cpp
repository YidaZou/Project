#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>

#include <adiak.hpp>

void array_fill_random(int *arr, int length) {
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = rand();
    }
}

void array_fill_sorted(int *arr, int length) {
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = i;
    }
}

void array_fill_reverseSorted(int *arr, int length) {
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = length - 1 - i;
    }
}

void array_fill_1perturbed(int *arr, int length) {
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
        if (i % 100 == 0)
            arr[i] = rand();
        else
            arr[i] = i;
    }
}

void dataInit(int *values, std::string inputType, int inputSize) {
    if (inputType == "Random") {
        array_fill_random(values, inputSize);
    } else if (inputType == "Sorted") {
        array_fill_sorted(values, inputSize);
    } else if (inputType == "ReverseSorted") {
        array_fill_reverseSorted(values, inputSize);
    } else if (inputType == "1Perturbed") {
        array_fill_1perturbed(values, inputSize);
    } else {
        printf("Error: Invalid input type\n");
        exit(0);
        return;
    }
}

// Check if sorted
void correctness_check(int arr[], int size) {
    int i;
    for (i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf("\nError: Array not sorted at indices %d and %d (values: %d and %d)\n",
                   i, i + 1, arr[i], arr[i + 1]);
        }
    }
    printf("\nArray sorted correctly\n");
    return;
}

static int comparable(const void *i, const void *j) {
    return (*(int *)i) - (*(int *)j);
}

void parallelSampleSort(int *localData, int localSize, int *pivots, int numProcs, int rank, MPI_Comm comm) {
    // Gather samples from all processes
    int *allSamples = NULL;
    int *localSamples = (int *)malloc(numProcs * sizeof(int));
    MPI_Gather(&localSize, 1, MPI_INT, localSamples, 1, MPI_INT, 0, comm);

    if (rank == 0) {
        allSamples = (int *)malloc(numProcs * localSize * sizeof(int));
    }

    MPI_Gatherv(localData, localSize, MPI_INT, allSamples, localSamples, localSamples, MPI_INT, 0, comm);

    // Sort the samples on root
    if (rank == 0) {
        std::sort(allSamples, allSamples + numProcs * localSize);
        // Choose pivots
        for (int i = 1; i < numProcs; ++i) {
            pivots[i - 1] = allSamples[i * localSize];
        }
        pivots[numProcs - 1] = allSamples[numProcs * localSize - 1];
    }

    // Broadcast pivots to all processes
    MPI_Bcast(pivots, numProcs - 1, MPI_INT, 0, comm);

    // Local partitioning using selected pivots
    int *localCounts = (int *)malloc(numProcs * sizeof(int));
    int *prefixCounts = (int *)malloc(numProcs * sizeof(int));
    int *localPartitions = (int *)malloc(localSize * sizeof(int));

    for (int i = 0; i < numProcs; ++i) {
        localCounts[i] = 0;
    }

    for (int i = 0; i < localSize; ++i) {
        int j = 0;
        while (j < numProcs - 1 && localData[i] >= pivots[j]) {
            ++j;
        }
        localCounts[j]++;
        localPartitions[i] = j;
    }

    // Compute prefix sum of counts
    prefixCounts[0] = 0;
    for (int i = 1; i < numProcs; ++i) {
        prefixCounts[i] = prefixCounts[i - 1] + localCounts[i - 1];
    }

    // Move data to correct partitions
    int *localCopy = (int *)malloc(localSize * sizeof(int));
    for (int i = 0; i < localSize; ++i) {
        int dest = prefixCounts[localPartitions[i]]++;
        localCopy[dest] = localData[i];
    }

    // Copy data back to localData
    memcpy(localData, localCopy, localSize * sizeof(int));

    free(localSamples);
    free(allSamples);
    free(localCounts);
    free(prefixCounts);
    free(localPartitions);
    free(localCopy);
}

int main(int argc, char **argv) {
    CALI_CXX_MARK_FUNCTION;
    CALI_MARK_BEGIN("main");

    cali::ConfigManager mgr;
    mgr.start();

    // Variable Declarations
    int numProcs, rank, root = 0;
    int countElements, countElementsLocal;
    int *input, *output;
    std::string inputType;
    MPI_Status status;

    // Initializing
    numProcs = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read input on root
    if (rank == root) {
        countElements = atoi(argv[2]);
        input = (int *)malloc(countElements * sizeof(int));
        if (input == NULL) {
            printf("Error: Can not allocate memory \n");
        }

        // Initialize data
        inputType = argv[3];
        dataInit(input, inputType, countElements);

        // Print original data
        printf("Original data on root:\n");
        for (int i = 0; i < countElements; i++) {
            printf("%d ", input[i]);
        }
        printf("\n");
    }

    // Broadcast the size of the array to all processes
    MPI_Bcast(&countElements, 1, MPI_INT, root, MPI_COMM_WORLD);

    // Allocate memory for local data
    countElementsLocal = countElements / numProcs;
    int *localData = (int *)malloc(countElementsLocal * sizeof(int));

    // Scatter data to all processes
    MPI_Scatter(input, countElementsLocal, MPI_INT, localData, countElementsLocal, MPI_INT, root, MPI_COMM_WORLD);

    // Sort local data using parallel SampleSort
    int *pivots = (int *)malloc((numProcs - 1) * sizeof(int));
    parallelSampleSort(localData, countElementsLocal, pivots, numProcs, rank, MPI_COMM_WORLD);

    // Gather sorted data on root
    if (rank == root) {
        output = (int *)malloc(countElements * sizeof(int));
    }
    MPI_Gather(localData, countElementsLocal, MPI_INT, output, countElementsLocal, MPI_INT, root, MPI_COMM_WORLD);

    // Print sorted data on root
    if (rank == root) {
        printf("Sorted data on root:\n");
        for (int i = 0; i < countElements; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");

        // Check correctness
        CALI_MARK_BEGIN("correctness_check");
        correctness_check(output, countElements);
        CALI_MARK_END("correctness_check");

        free(input);
        free(output);
    }

    // Finalize
    MPI_Finalize();

    adiak::init(NULL);
    adiak::launchdate();                    // launch date of the job
    adiak::libraries();                     // Libraries used
    adiak::cmdline();                       // Command line used to launch the job
    adiak::clustername();                   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");        // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", countElements);      // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);         // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcs);           // The number of processors (MPI ranks)
    adiak::value("group_num", 6);                  // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    CALI_MARK_END("main");

    mgr.stop();
    mgr.flush();

    return 0;
}

// Implementation help from https://github.com/peoro/Parasort
