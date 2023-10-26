# CSCE 435 Group project

## 1. Group members:
1. Yida Zou
2. Brian Chen
3. Third
4. Fourth

---

## 2. _due 10/25_ Project topic

We chose to use the suggested topic idea:
Choose 3+ parallel sorting algorithms, implement in MPI and CUDA.  Examine and compare performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core)

We will communicate via iMessage.

Algorithms we will implement:

- Sample Sort (MPI + CUDA)
```
function SampleSort(unsortedList, t):  //t = thread count
    //Divide input into samples depending on number of threads
    sampleSize = calculateSampleSize(unsortedList, t)
    sample = selectSample(unsortedList, sampleSize)
    
    //Distribute the sample to all processors 
    distribute(sample)  //Using MPI
    
    //Each thread sorts the given sample locally
    sortedSample = sortLocally(sample)    //sort using cudaMemcpy

    //Gather sorted samples
    sortedSamples = communicate(sample)    //Using MPI

    //Merge and sort all samples together
    sortedLists = mergeSublists(sortedSamples)
    
    return sortedSublist
```

- Merge Sort (MPI + CUDA)
```
function void merge_sort(arr, numThreads):

    # Initialize MPI environment
    MPI_Init()

    # Get the MPI rank and size
    rank = MPI_Comm_rank(MPI_COMM_WORLD)
    size = MPI_Comm_size(MPI_COMM_WORLD)

    # Divide and distribute data across nodes
    local_data = distribute_data(rank, size)

    # Sort local data using CUDA
    local_data = cuda_merge_sort(local_data)

    # Communicate data between nodes, and merge lists
    sorted_data = gather_and_merge(local_data, rank, size)

    # Finalize MPI environment
    MPI_Finalize()

    return sorted_data
```
To vary our algorithms, we will apply the following communication and parallelization strategies to our merge sort:
- fork/join parallelism
- point-to-point communication

- Bitonic Sort (MPI + CUDA)
```
function void bitonic_sort(arr, numThreads) {

    MPI_Init(...);
    rank = MPI_Comm_rank(...);
    size = MPI_Comm_size(...);

    MPI_Scatter(...);

    for (...) { // major step
        for (...) { // minor step
            sortOnGPU(...);
        }
    }

    MPI_Gather(...);
    MPI_Finalize();
    return sorted_data;
}

```

