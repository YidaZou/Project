#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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
	int perturb = length / 100;
	for (i = 0; i < length; ++i) {
		if (i % perturb == 0)
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
			return;
		}
	}
	printf("\nArray sorted correctly\n");
	return;
}

static int comparable(const void *i, const void *j) {
	return (*(int *)i) - (*(int *)j);
}

int main(int argc, char** argv) {
	// Variable Declarations
	int numProcs, rank, root = 0;
	int i, j, k, countElements, countElementsLocal, countElementsToSort;
	int count, temp;
	int *arrayGlobal, *arrayLocal;
	int *splitterLocal, *splitterGlobal;
	int *bucketGlobal, *bucketLocalBuf, *bucketLocal;
	int *outputBuf;
	std::string inputType;
	MPI_Status status;

	// Initializing
	numProcs = atoi(argv[1]);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	CALI_CXX_MARK_FUNCTION;
	CALI_MARK_BEGIN("main");

	cali::ConfigManager mgr;
	mgr.start();

	/**** Reading Input ****/
	//printf("Rank %d: Reading Input\n", rank);

	if (rank == root){

		countElements = atoi(argv[2]);
		arrayGlobal = (int *) malloc (countElements*sizeof(int));
		//printf("Memory: %d:\n", countElements*sizeof(int));
		if(arrayGlobal == NULL) {
			printf("Error : Can not allocate memory \n");
		}

		/* Initialise random number generator  */
		inputType = argv[3];
		dataInit(arrayGlobal, inputType, countElements);
		printf ( "\n\n ");
	}

	/**** Sending Data ****/
	//printf("Rank %d: Sending Data\n", rank);
	MPI_Bcast (&countElements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(( countElements % numProcs) != 0){
			if(rank == root)
			printf("Number of Elements are not divisible by Numprocs \n");
				MPI_Finalize();
			exit(0);
	}

	countElementsLocal = countElements / numProcs;
	arrayLocal = (int *) malloc (countElementsLocal * sizeof (int));
	//printf("Memory: %d:\n", countElementsLocal*sizeof(int));

	MPI_Scatter(arrayGlobal, countElementsLocal, MPI_INT, arrayLocal,
					countElementsLocal, MPI_INT, root, MPI_COMM_WORLD);

	/**** Sorting Locally ****/
	//printf("Rank %d: Sorting Locally\n", rank);
	qsort ((char *) arrayLocal, countElementsLocal, sizeof(int), comparable);

	/**** Choosing Local Splitters ****/
	//printf("Rank %d: Choosing Local Splitters\n", rank);
	splitterLocal = (int *) malloc (sizeof (int) * (numProcs-1));
	//printf("Memory: %d:\n", (numProcs - 1)*sizeof(int));
	for (i=0; i< (numProcs-1); i++){
			splitterLocal[i] = arrayLocal[countElements/(numProcs*numProcs) * (i+1)];
	}

	/**** Gathering Local Splitters at Root ****/
	//printf("Rank %d: Gatherin Local Splitters at Root\n", rank);
	splitterGlobal = (int *) malloc (sizeof (int) * numProcs * (numProcs-1));
	//printf("Memory: %d:\n", numProcs * (numProcs - 1)*sizeof(int));
	MPI_Gather (splitterLocal, numProcs-1, MPI_INT, splitterGlobal, numProcs-1, MPI_INT, root, MPI_COMM_WORLD);

	/**** Choosing Global Splitters ****/
	if (rank == root){
		//printf("Rank %d: Choosing Global Splitters\n", rank);
		qsort ((char *) splitterGlobal, numProcs*(numProcs-1), sizeof(int), comparable);

		for (i=0; i<numProcs-1; i++)
		splitterLocal[i] = splitterGlobal[(numProcs-1)*(i+1)];
	}

	/**** Broadcasting Global Splitters ****/
	//printf("Rank %d: Broadcasting Global Splitters\n", rank);
	MPI_Bcast (splitterLocal, numProcs-1, MPI_INT, 0, MPI_COMM_WORLD);

	/**** Creating Buckets locally ****/
	//printf("Rank %d: Creating Buckets locally\n", rank);
	bucketGlobal = (int *) malloc (sizeof (int) * (countElements + numProcs));
	//printf("Memory: %d:\n", (countElements + numProcs) *sizeof(int));

	j = 0;
	k = 1;

	for (i=0; i<countElementsLocal; i++){
		if(j < (numProcs-1)){
		if (arrayLocal[i] < splitterLocal[j])
				bucketGlobal[((countElementsLocal + 1) * j) + k++] = arrayLocal[i];
		else{
			bucketGlobal[(countElementsLocal + 1) * j] = k-1;
				k=1;
				j++;
				i--;
		}
		}
		else
		bucketGlobal[((countElementsLocal + 1) * j) + k++] = arrayLocal[i];
	}
	bucketGlobal[(countElementsLocal + 1) * j] = k - 1;

	/**** Sending buckets to respective processors ****/
	//printf("Rank %d: Sending buckets to respective processors\n", rank);

	bucketLocalBuf = (int *) malloc (sizeof (int) * (countElements + numProcs));
	//printf("Memory: %d:\n", (countElements + numProcs)*sizeof(int));

	MPI_Alltoall (bucketGlobal, countElementsLocal + 1, MPI_INT, bucketLocalBuf,
						countElementsLocal + 1, MPI_INT, MPI_COMM_WORLD);

	/**** Rearranging BucketBuffer ****/
	//printf("Rank %d: Rearranging BucketBuffer\n", rank);
	bucketLocal = (int *) malloc (sizeof (int) * 2 * countElements / numProcs);
	//printf("Memory: %d:\n", (2 * countElementsLocal / numProcs)*sizeof(int));

	count = 1;

	for (j=0; j<numProcs; j++) {
	k = 1;
		for (i=0; i<bucketLocalBuf[(countElements/numProcs + 1) * j]; i++)
		bucketLocal[count++] = bucketLocalBuf[(countElements/numProcs + 1) * j + k++];
	}
	bucketLocal[0] = count-1;

	/**** Sorting Local Buckets using Bubble Sort ****/
	//printf("Rank %d: Sorting Local Buckets using Bubble Sort\n", rank);
	/*qsort ((char *) InputData, NoofElements_Bloc, sizeof(int), comparable); */

	countElementsToSort = bucketLocal[0];
	qsort ((char *) &bucketLocal[1], countElementsToSort, sizeof(int), comparable);

	/**** Gathering sorted sub blocks at root ****/
	if(rank == root) {
		//printf("Rank %d: Gathering sorted sub blocks at root\n", rank);
		outputBuf = (int *) malloc (sizeof(int) * 2 * countElements);
		//printf("Memory: %d:\n", 2 * countElements*sizeof(int));
	}

	MPI_Gather (bucketLocal, 2 * countElementsLocal, MPI_INT, outputBuf, 2 * countElementsLocal, MPI_INT, root, MPI_COMM_WORLD);

	/**** Rearranging output buffer ****/
	if (rank == root){
		//printf("Rank %d: Rearranging output buffer\n", rank);
		count = 0;
		for(j=0; j<numProcs; j++){
		k = 1;
		for(i=0; i<outputBuf[(2 * countElements/numProcs) * j]; i++)
				arrayGlobal[count++] = outputBuf[(2*countElements/numProcs) * j + k++];
		}

		/**** Printng the output ****/
		// //printf("Rank %d: Printing the output\n", rank);
		// printf ( "Number of Elements to be sorted : %d \n", countElements);
		// printf( "Sorted output sequence is\n\n");
		// for (i=0; i<countElements; i++){
		// 	printf( "%d   ", arrayGlobal[i]);
		// }

		CALI_MARK_BEGIN("correctness_check");
		correctness_check(arrayGlobal, countElements);
		CALI_MARK_END("correctness_check");

		printf ( " \n " );
		free(arrayGlobal);
		free(outputBuf);
	}/* MyRank==0*/

		free(arrayLocal);
		free(splitterLocal);
		free(splitterGlobal);
		free(bucketGlobal);
		free(bucketLocalBuf);
		free(bucketLocal);

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

// Flush Caliper output before finalizing MPI
	mgr.stop();
	mgr.flush();

	MPI_Finalize();

	return 0;
}

// Implementation help from https://github.com/peoro/Parasort
