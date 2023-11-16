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
    } else if (inputType == "1perturbed") {
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
    // Local partitioning and sorting
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");
    std::sort(localData, localData + localSize);
	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");


    // Choosing the Splitters
    std::vector<int> localSplitters(numProcs - 1);
	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_small");
    for (int i = 1; i < numProcs; ++i) {
        localSplitters[i - 1] = localData[i * (localSize / numProcs)];
    }
	CALI_MARK_END("comp_small");
	CALI_MARK_END("comp");

    // Gather all local splitters at root (rank 0)
    std::vector<int> allSplitters(numProcs * (numProcs - 1));
	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(localSplitters.data(), numProcs - 1, MPI_INT,
               allSplitters.data(), numProcs - 1, MPI_INT, 0, comm);
	CALI_MARK_END("MPI_Gather");
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");


    // Completing the sort

    if (rank == 0) {
        // Sort the gathered splitters
		CALI_MARK_BEGIN("comp");
		CALI_MARK_BEGIN("comp_small");
        std::sort(allSplitters.begin(), allSplitters.end());

        // Choose pivots
        for (int i = 1; i < numProcs; ++i) {
            pivots[i - 1] = allSplitters[i * (numProcs - 1)];
        }
        pivots[numProcs - 1] = allSplitters[numProcs * (numProcs - 1) - 1];
		CALI_MARK_END("comp_small");
		CALI_MARK_END("comp");
    }

    // Broadcast pivots to all processes
	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(pivots, numProcs - 1, MPI_INT, 0, comm);
	CALI_MARK_END("MPI_Bcast");
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");

    // Local partitioning using selected pivots
    std::vector<int> localCounts(numProcs, 0);
    std::vector<int> prefixCounts(numProcs, 0);
    std::vector<int> localPartitions(localSize);

	CALI_MARK_BEGIN("comp");
	CALI_MARK_BEGIN("comp_large");

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
    std::vector<int> localCopy(localSize);
    for (int i = 0; i < localSize; ++i) {
        int dest = prefixCounts[localPartitions[i]]++;
        localCopy[dest] = localData[i];
    }

    // Copy data back to localData
    std::memcpy(localData, localCopy.data(), localSize * sizeof(int));

	CALI_MARK_END("comp_large");
	CALI_MARK_END("comp");
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
		CALI_MARK_BEGIN("data_init");
        dataInit(input, inputType, countElements);
		CALI_MARK_END("data_init");

        // Print original data
        // printf("Original data on root:\n");
        // for (int i = 0; i < countElements; i++) {
        //     printf("%d ", input[i]);
        // }
        // printf("\n");
    }

	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_small");
	CALI_MARK_BEGIN("MPI_Bcast");
    // Broadcast the size of the array to all processes
    MPI_Bcast(&countElements, 1, MPI_INT, root, MPI_COMM_WORLD);
	CALI_MARK_END("MPI_Bcast");
	CALI_MARK_END("comm_small");
	CALI_MARK_END("comm");

    // Allocate memory for local data
    countElementsLocal = countElements / numProcs;
    int *localData = (int *)malloc(countElementsLocal * sizeof(int));

    // Scatter data to all processes

	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_large");
	CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(input, countElementsLocal, MPI_INT, localData, countElementsLocal, MPI_INT, root, MPI_COMM_WORLD);
	CALI_MARK_END("MPI_Scatter");
	CALI_MARK_END("comm_large");
	CALI_MARK_END("comm");

    // Sort local data using parallel SampleSort
    int *pivots = (int *)malloc((numProcs - 1) * sizeof(int));
    parallelSampleSort(localData, countElementsLocal, pivots, numProcs, rank, MPI_COMM_WORLD);

    // Gather sorted data on root
    if (rank == root) {
        output = (int *)malloc(countElements * sizeof(int));
    }

	CALI_MARK_BEGIN("comm");
	CALI_MARK_BEGIN("comm_large");
	CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(localData, countElementsLocal, MPI_INT, output, countElementsLocal, MPI_INT, root, MPI_COMM_WORLD);
	CALI_MARK_END("MPI_Gather");
	CALI_MARK_END("comm_large");
	CALI_MARK_END("comm");

    // Print sorted data on root
    if (rank == root) {
        // printf("Sorted data on root:\n");
        // for (int i = 0; i < countElements; i++) {
        //     printf("%d ", output[i]);
        // }
        // printf("\n");

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
    adiak::value("implementation_source", "Online+AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    CALI_MARK_END("main");

    mgr.stop();
    mgr.flush();

    return 0;
}

// Implementation help from https://github.com/peoro/Parasort
