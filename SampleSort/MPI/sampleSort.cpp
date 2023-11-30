#include "mpi.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <random>
#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>

#define ROOT 0

void array_fill_random(std::vector<int>& arr) {
    srand(time(NULL));
    for (int i = 0; i < arr.size(); ++i) {
        arr[i] = rand();
    }
}

void array_fill_sorted(std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); ++i) {
        arr[i] = i;
    }
}

void array_fill_reverseSorted(std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); ++i) {
        arr[i] = arr.size() - 1 - i;
    }
}

void array_fill_1perturbed(std::vector<int>& arr) {
    srand(time(NULL));
    for (int i = 0; i < arr.size(); i++) {
        if (i % 100 == 0)
            arr[i] = rand();
        else
            arr[i] = i;
    }
}

void dataInit(std::vector<int>& values, const std::string& inputType, int arraySize) {
	values.resize(arraySize);
    if (inputType == "Random") {
        array_fill_random(values);
    } else if (inputType == "Sorted") {
        array_fill_sorted(values);
    } else if (inputType == "ReverseSorted") {
        array_fill_reverseSorted(values);
    } else if (inputType == "1Perturbed") {
        array_fill_1perturbed(values);
    } else {
        printf("Error: Invalid input type\n");
        exit(0);
        return;
    }
}

void correctness_check(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size() - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf("\nError: Array not sorted at indices %d and %d (values: %d and %d)\n",
                i, i + 1, arr[i], arr[i + 1]);
            return;
        }
    }
    printf("\nArray sorted correctly\n");
    return;
}

// Function to perform sample sort on local data
std::vector<int> sampleSortHelper(std::vector<int>& dataLocal, int rank, int numProcs, int sizeGlobal) {

	// Declare variables
	int dataSortedLocalSize;
	int dataIncomingGlobal;

	std::vector<int> samplesLocal;
	std::vector<int> samplesGlobal;
	std::vector<int> dataSorted;
	std::vector<int> dataSortedGlobal;

	std::vector<int> splitters(numProcs - 1);
	std::vector<std::vector<int>> buckets(numProcs);

	std::vector<int> dataOutgoing;
    std::vector<int> dataOutgoingCounts(numProcs, 0);
	std::vector<int> dataIncomingCounts(numProcs);

	std::vector<int> offsets(numProcs, 0);
	std::vector<int> offsetsOutgoing(numProcs, 0);
    std::vector<int> offsetsIncoming(numProcs, 0);

	// Step 1: Local sorting
    std::sort(dataLocal.begin(), dataLocal.end());

    // Step 2: Selecting samples for splitters
    for (int i = 1; i < numProcs; i++) {
        // Choose samples at regular intervals for efficient splitting
        int index = (dataLocal.size() / numProcs) * i;
        samplesLocal.push_back(dataLocal[index]);
    }

    // Step 3: Gather samples at root and pick splitters
    if (rank == ROOT) {
        samplesGlobal.resize(numProcs * (numProcs - 1));
    }
    MPI_Gather(samplesLocal.data(), samplesLocal.size(), MPI_INT, samplesGlobal.data(), samplesLocal.size(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Sort the gathered samples and select splitters
        std::sort(samplesGlobal.begin(), samplesGlobal.end());
        for (int i = 0; i < numProcs - 1; i++) {
            splitters[i] = samplesGlobal[(i * numProcs) + (numProcs / 2)];
        }
    }

    MPI_Bcast(splitters.data(), splitters.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // Step 4: Bucketing based on splitters
    for (int i = 0; i < dataLocal.size(); i++) {
        // Determine the bucket index for each value
        int bucketIdx = std::lower_bound(splitters.begin(), splitters.end(), dataLocal[i]) - splitters.begin();
        buckets[bucketIdx].push_back(dataLocal[i]);
    }

    // Step 5: Prepare data for sending
    for (int i = 0; i < numProcs; i++) {
        // Concatenate data from buckets for sending
        dataOutgoing.insert(dataOutgoing.end(), buckets[i].begin(), buckets[i].end());
        dataOutgoingCounts[i] = buckets[i].size();
    }

    // Step 6: Communicate the sizes of the buckets
    MPI_Alltoall(dataOutgoingCounts.data(), 1, MPI_INT, dataIncomingCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Step 7: Calculate offsets for data exchange
    for (int i = 1; i < numProcs; i++) {
        offsetsOutgoing[i] = dataOutgoingCounts[i - 1] + offsetsOutgoing[i - 1];
        offsetsIncoming[i] = dataIncomingCounts[i - 1] + offsetsIncoming[i - 1];
    }

    // Step 8: Gather the sorted data after exchanging
    dataIncomingGlobal = std::accumulate(dataIncomingCounts.begin(), dataIncomingCounts.end(), 0);
   	dataSorted.resize(dataIncomingGlobal);
    MPI_Alltoallv(dataOutgoing.data(), dataOutgoingCounts.data(), offsetsOutgoing.data(), MPI_INT, dataSorted.data(), dataIncomingCounts.data(), offsetsIncoming.data(), MPI_INT, MPI_COMM_WORLD);

    // Step 9: Final local sort on the exchanged data
    std::sort(dataSorted.begin(), dataSorted.end());

    // Step 10: Prepare for gathering sorted data at the root
    dataSortedLocalSize = dataSorted.size();
    MPI_Gather(&dataSortedLocalSize, 1, MPI_INT, dataIncomingCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 11: Calculate offsets for gathering
    if (rank == ROOT) {
        for (int i = 1; i < numProcs; i++) {
            offsets[i] = dataIncomingCounts[i - 1] + offsets[i - 1];
        }
        dataSortedGlobal.resize(sizeGlobal);
    }

    // Step 12: Gather sorted data at root
    MPI_Gatherv(dataSorted.data(), dataSorted.size(), MPI_INT, dataSortedGlobal.data(), dataIncomingCounts.data(), offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);

	return dataSortedGlobal;
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN("main");

	// Configure the Caliper profiler
	cali::ConfigManager mgr;
	mgr.start();

	// Declare variables
	int sizeLocal;
	int sizeGlobal;

	std::string inputType;

	std::vector<int> dataLocal;
	std::vector<int> dataGlobal;

	// Parse command line arguments
	sizeGlobal = atoi(argv[1]);
	inputType = argv[2];

	// Initialize MPI
	int numProcs, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Initialize ADIAK for performance measurement
	adiak::init(NULL);
	adiak::launchdate();
	adiak::libraries();
	adiak::cmdline();
	adiak::clustername();
	adiak::value("Algorithm", "SampleSort");
	adiak::value("ProgrammingModel", "MPI");
	adiak::value("Datatype", "int");
	adiak::value("SizeOfDatatype", sizeof(int));
	adiak::value("InputSize", sizeGlobal);
	adiak::value("InputType", inputType);
	adiak::value("num_procs", numProcs);
	adiak::value("group_num", 6);
	adiak::value("implementation_source", "AI");

	// Read and initialize data at the root process
	if (rank == ROOT) {
		CALI_MARK_BEGIN("data_init");
		dataInit(dataGlobal, inputType, sizeGlobal);
		CALI_MARK_END("data_init");
	}

	// Calculate local data size and resize the local vector
	sizeLocal = sizeGlobal / numProcs;
	dataLocal.resize(sizeLocal);

	// Scatter the global data to all processes
	MPI_Scatter(dataGlobal.data(), sizeLocal, MPI_INT, dataLocal.data(), sizeLocal, MPI_INT, 0, MPI_COMM_WORLD);

	// Perform sample sort on local data and get the sorted global data
	dataGlobal = sampleSortHelper(dataLocal, rank, numProcs, sizeGlobal);

	// Check the correctness of the sorted data at the root process
	if (rank == ROOT) {
		CALI_MARK_BEGIN("correctness_check");
		correctness_check(dataGlobal);
		CALI_MARK_END("correctness_check");
	}

	CALI_MARK_END("main");

	// Finalize MPI
	MPI_Finalize();

	// Return from the main function
	return 0;
}
