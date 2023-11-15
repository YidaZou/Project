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
    } else if (inputType == "1perturbed") {
        array_fill_1perturbed(values, inputSize);
    } else {
        printf("Error: Invalid input type\n");
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
    std::string input_type;
    input_type = argv[3];

    CALI_CXX_MARK_FUNCTION;
    CALI_MARK_BEGIN("main");

    cali::ConfigManager mgr;
    mgr.start();

    // Variable Declarations
    int Numprocs, MyRank, Root = 0;
    int i, j, k, NoofElements, NoofElements_Bloc, NoElementsToSort;
    int count, temp;
    int *Input, *InputData;
    int *Splitter, *AllSplitter;
    int *Buckets, *BucketBuffer, *LocalBucket;
    int *OutputBuffer, *Output;
    FILE *InputFile, *fp;
    MPI_Status status;

    // Initializing
    Numprocs = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    // Reading Input
    if (MyRank == Root) {
        NoofElements = atoi(argv[2]);
        Input = (int *)malloc(NoofElements * sizeof(int));
        if (Input == NULL)
            printf("Error : Can not allocate memory for Input\n");
        Output = (int *)malloc(NoofElements * sizeof(int));
        if (Output == NULL)
            printf("Error : Can not allocate memory for Output\n");

        dataInit(Input, input_type, NoofElements);
    } else {
        Input = (int *)malloc(sizeof(int));
        if (Input == NULL)
            printf("Error : Can not allocate memory for Input\n");
        Output = (int *)malloc(sizeof(int));
        if (Output == NULL)
            printf("Error : Can not allocate memory for Output\n");
    }

    // Broadcasting NoofElements
    MPI_Bcast(&NoofElements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if ((NoofElements % Numprocs) != 0) {
        if (MyRank == Root)
            printf("Number of Elements are not divisible by Numprocs\n");
        MPI_Finalize();
        exit(0);
    }

    NoofElements_Bloc = NoofElements / Numprocs;
    InputData = (int *)malloc(NoofElements_Bloc * sizeof(int));

    MPI_Scatter(Input, NoofElements_Bloc, MPI_INT, InputData, NoofElements_Bloc, MPI_INT, Root, MPI_COMM_WORLD);

    // Sorting Locally
    printf("Rank %d: Before local sort\n", MyRank);
    qsort((char *)InputData, NoofElements_Bloc, sizeof(int), comparable);
    printf("Rank %d: After local sort\n", MyRank);

    // Choosing Local Splitters
    Splitter = (int *)malloc(sizeof(int) * (Numprocs - 1));
    for (i = 0; i < (Numprocs - 1); i++) {
        Splitter[i] = InputData[NoofElements / (Numprocs * Numprocs) * (i + 1)];
    }

    // Gathering Local Splitters at Root
    AllSplitter = (int *)malloc(sizeof(int) * Numprocs * (Numprocs - 1));
    MPI_Gather(Splitter, Numprocs - 1, MPI_INT, AllSplitter, Numprocs - 1, MPI_INT, Root, MPI_COMM_WORLD);

    // Choosing Global Splitters
    if (MyRank == Root) {
        printf("Root: Before global sort of splitters\n");
        qsort((char *)AllSplitter, Numprocs * (Numprocs - 1), sizeof(int), comparable);
        printf("Root: After global sort of splitters\n");

        for (i = 0; i < Numprocs - 1; i++)
            Splitter[i] = AllSplitter[(Numprocs - 1) * (i + 1)];
    }

    // Broadcasting Global Splitters
    MPI_Bcast(Splitter, Numprocs - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Creating Numprocs Buckets locally
    Buckets = (int *)malloc(sizeof(int) * (NoofElements + Numprocs));

    j = 0;
    k = 1;

    for (i = 0; i < NoofElements_Bloc; i++) {
        if (j < (Numprocs - 1)) {
            if (InputData[i] < Splitter[j])
                Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i];
            else {
                Buckets[(NoofElements_Bloc + 1) * j] = k - 1;
                k = 1;
                j++;
                i--;
            }
        } else
            Buckets[((NoofElements_Bloc + 1) * j) + k++] = InputData[i];
    }
    Buckets[(NoofElements_Bloc + 1) * j] = k - 1;

    // Sending buckets to respective processors
    BucketBuffer = (int *)malloc(sizeof(int) * (NoofElements + Numprocs));

    MPI_Alltoall(Buckets, NoofElements_Bloc + 1, MPI_INT, BucketBuffer, NoofElements_Bloc + 1, MPI_INT, MPI_COMM_WORLD);

    // Rearranging BucketBuffer
    LocalBucket = (int *)malloc(sizeof(int) * 2 * NoofElements / Numprocs);

    count = 1;

    for (j = 0; j < Numprocs; j++) {
        k = 1;
        for (i = 0; i < BucketBuffer[(NoofElements / Numprocs + 1) * j]; i++)
            LocalBucket[count++] = BucketBuffer[(NoofElements / Numprocs + 1) * j + k++];
    }
    LocalBucket[0] = count - 1;

    // Sorting Local Buckets using Bubble Sort
    // qsort((char *)InputData, NoofElements_Bloc, sizeof(int), comparable);

    NoElementsToSort = LocalBucket[0];
    printf("Rank %d: Before local sort of buckets\n", MyRank);
    qsort((char *)&LocalBucket[1], NoElementsToSort, sizeof(int), comparable);
    printf("Rank %d: After local sort of buckets\n", MyRank);

    // Gathering sorted sub blocks at root
    if (MyRank == Root) {
        OutputBuffer = (int *)malloc(sizeof(int) * 2 * NoofElements);
        Output = (int *)malloc(sizeof(int) * NoofElements);
    }

    MPI_Gather(LocalBucket, 2 * NoofElements_Bloc, MPI_INT, OutputBuffer, 2 * NoofElements_Bloc, MPI_INT, Root, MPI_COMM_WORLD);

    // Rearranging output buffer
    if (MyRank == Root) {
        count = 0;
        for (j = 0; j < Numprocs; j++) {
            k = 1;
            for (i = 0; i < OutputBuffer[(2 * NoofElements / Numprocs) * j]; i++)
                Output[count++] = OutputBuffer[(2 * NoofElements / Numprocs) * j + k++];
        }

        // Printing the output
        if ((fp = fopen("sort.out", "w")) == NULL) {
            printf("Can't Open Output File\n");
            exit(0);
        }

        fprintf(fp, "Number of Elements to be sorted : %d\n", NoofElements);
        printf("Number of Elements to be sorted : %d\n", NoofElements);
        fprintf(fp, "The sorted sequence is :\n");
        printf("Sorted output sequence is\n\n");
        for (i = 0; i < NoofElements; i++) {
            fprintf(fp, "%d\n", Output[i]);
            printf("%d   ", Output[i]);
        }
        printf("\n");
        fclose(fp);

        CALI_MARK_BEGIN("correctness_check");
        correctness_check(Output, NoofElements);
        CALI_MARK_END("correctness_check");

        free(Input);
        free(OutputBuffer);
        free(Output);
    } /* MyRank==0*/

    free(InputData);
    free(Splitter);
    free(AllSplitter);
    free(Buckets);
    free(BucketBuffer);
    free(LocalBucket);

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
    adiak::value("InputSize", NoofElements);      // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type);         // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", Numprocs);           // The number of processors (MPI ranks)
    adiak::value("group_num", 6);                  // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    CALI_MARK_END("main");
    return 0;
}

// Implementation help from https://github.com/peoro/Parasort
