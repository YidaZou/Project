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
    } else if (inputType == "1perturbed") {
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
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::sort(dataLocal.begin(), dataLocal.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Step 2: Selecting samples for splitters
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    for (int i = 1; i < numProcs; i++) {
        // Choose samples at regular intervals for efficient splitting
        int index = (dataLocal.size() / numProcs) * i;
        samplesLocal.push_back(dataLocal[index]);
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Step 3: Gather samples at root
    if (rank == ROOT) {
        samplesGlobal.resize(numProcs * (numProcs - 1));
    }
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(samplesLocal.data(), samplesLocal.size(), MPI_INT, samplesGlobal.data(), samplesLocal.size(), MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    if (rank == 0) {
        // Sort the gathered samples and select splitters
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_small");
        std::sort(samplesGlobal.begin(), samplesGlobal.end());
        for (int i = 0; i < numProcs - 1; i++) {
            splitters[i] = samplesGlobal[(i * numProcs) + (numProcs / 2)];
        }
        CALI_MARK_END("comp_small");
        CALI_MARK_END("comp");
    }

    // Step 4: Broadcast splitters to all processes
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(splitters.data(), splitters.size(), MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Step 5: Bucketing based on splitters
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < dataLocal.size(); i++) {
        // Determine the bucket index for each value
        int bucketIdx = std::lower_bound(splitters.begin(), splitters.end(), dataLocal[i]) - splitters.begin();
        buckets[bucketIdx].push_back(dataLocal[i]);
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Step 6: Prepare data for sending
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (int i = 0; i < numProcs; i++) {
        // Concatenate data from buckets for sending
        dataOutgoing.insert(dataOutgoing.end(), buckets[i].begin(), buckets[i].end());
        dataOutgoingCounts[i] = buckets[i].size();
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Step 7: Communicate the sizes of the buckets
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Alltoall");
    MPI_Alltoall(dataOutgoingCounts.data(), 1, MPI_INT, dataIncomingCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Alltoall");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Step 8: Calculate offsets for data exchange
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    for (int i = 1; i < numProcs; i++) {
        offsetsOutgoing[i] = dataOutgoingCounts[i - 1] + offsetsOutgoing[i - 1];
        offsetsIncoming[i] = dataIncomingCounts[i - 1] + offsetsIncoming[i - 1];
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Step 9: Gather the exchanged data
    dataIncomingGlobal = std::accumulate(dataIncomingCounts.begin(), dataIncomingCounts.end(), 0);
   	dataSorted.resize(dataIncomingGlobal);
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Alltoall");
    MPI_Alltoallv(dataOutgoing.data(), dataOutgoingCounts.data(), offsetsOutgoing.data(), MPI_INT, dataSorted.data(), dataIncomingCounts.data(), offsetsIncoming.data(), MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Alltoall");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Step 10: Final local sort on the exchanged data
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    std::sort(dataSorted.begin(), dataSorted.end());
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Step 11: Prepare for gathering of sorted data at the root
    dataSortedLocalSize = dataSorted.size();
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(&dataSortedLocalSize, 1, MPI_INT, dataIncomingCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    // Step 12: Calculate offsets for gathering
    if (rank == ROOT) {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_small");
        for (int i = 1; i < numProcs; i++) {
            offsets[i] = dataIncomingCounts[i - 1] + offsets[i - 1];
        }
        dataSortedGlobal.resize(sizeGlobal);
        CALI_MARK_END("comp_small");
        CALI_MARK_END("comp");
    }

    // Step 13: Gather sorted data at root
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gatherv(dataSorted.data(), dataSorted.size(), MPI_INT, dataSortedGlobal.data(), dataIncomingCounts.data(), offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

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
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
	MPI_Scatter(dataGlobal.data(), sizeLocal, MPI_INT, dataLocal.data(), sizeLocal, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

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
