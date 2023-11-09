#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <caliper/cali-manager.h>
#include <caliper/cali.h>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

// TODO: change cmake and mpi.grace_job files to use new file name

void bitonic_merge(int* arr, int start, int size, int dir) {
    if (size > 1) {
        int k = size / 2;
        for (int i = start; i < start + k; ++i) {
            int j = i + k;
            if ((arr[i] > arr[j] && dir == 1) || (arr[i] < arr[j] && dir == 0)) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        bitonic_merge(arr, start, k, dir);
        bitonic_merge(arr, start + k, k, dir);
    }
}

void bitonic_sort(int* arr, int start, int size, int dir) {
    if (size > 1) {
        int k = size / 2;
        bitonic_sort(arr, start, k, 1);
        bitonic_sort(arr, start + k, k, 0);
        bitonic_merge(arr, start, size, dir);
    }
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    int sizeOfArray;
    if (argc == 2) {
        sizeOfArray = atoi(argv[1]);
    } else {
        printf("\n Please provide the size of the array");
        return 0;
    }

    int numtasks, taskid, numworkers, source, dest, mtype, arr[sizeOfArray];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    
    // TODO: define const char* variables for caliper region names 
    CALI_MARK_BEGIN("Initialization");

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
        exit(1);
    }

    numworkers = numtasks - 1;

    cali::ConfigManager mgr;
    mgr.start();    

    if (taskid == MASTER) {
        // Generate random data
        srand(time(NULL));
        for (int i = 0; i < sizeOfArray; ++i) {
            arr[i] = rand() % 100;
        }

        // Print unsorted array
        printf("Unsorted array: ");
        for (int i = 0; i < sizeOfArray; ++i) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        // Distribute data to worker tasks
        int averow = sizeOfArray / numworkers;
        int extra = sizeOfArray % numworkers;
        int offset = 0;
        mtype = FROM_MASTER;

        CALI_MARK_BEGIN("Data_Distribution");

        for (dest = 1; dest <= numworkers; dest++) {
            int rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&arr[offset], rows, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }

        CALI_MARK_END("Data_Distribution");
    }

    if (taskid > MASTER) {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&mtype, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&arr, sizeOfArray, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        CALI_MARK_BEGIN("Bitonic_Sort");

        bitonic_sort(arr, offset, sizeOfArray, 1);

        CALI_MARK_END("Bitonic_Sort");

        // Send sorted data back to master
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&sizeOfArray, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&arr, sizeOfArray, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }

    if (taskid == MASTER) {
        mtype = FROM_WORKER;
        for (int i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&sizeOfArray, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&arr[offset], sizeOfArray, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
        }

        // Print sorted array
        printf("Sorted array: ");
        for (int i = 0; i < sizeOfArray; ++i) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    CALI_MARK_END("Initialization");
    mgr.stop();
    mgr.flush();
    MPI_Finalize();
    return 0;
}
