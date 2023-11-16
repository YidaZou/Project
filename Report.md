# CSCE 435 Group project

## 0. Group number:
6
## 1. Group members:
1. Yida Zou
2. Brian Chen
3. Sam Hollenbeck
4. Alex Pantazopol

## 2. Project topic

We chose to use the suggested topic idea:

Choose 3+ parallel sorting algorithms, implement in MPI and CUDA.  Examine and compare performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

## 2. Project description

We will communicate via iMessage.

Algorithms we will implement:

- Sample Sort (MPI)
- Sample Sort (CUDA)
- Merge Sort (MPI)
- Merge Sort (CUDA)
- Bitonic Sort (MPI)
- Bitonic Sort (CUDA)

We will compare the performance of these three algorithms with the metrics stated in the project topic.

To vary our algorithms, we will apply the following communication and parallelization strategies:
- fork/join parallelism
- point-to-point communication


## 3. Pseudocode
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
## 3. Evaluation plan
Some of the things we will compare are:

Runtimes between parallel sorting algorithms on GPU and CPU (MPI vs CUDA)
Scaling the number of threads or processors
Scaling the problem size (length of array to sort)

## 3. Project implementation

Currently, we only have graphs generated of Random InputType for our 6 algorithms.

## 4. Performance evaluation

Include detailed analysis of computation performance, communication performance.
Include figures and explanation of your analysis.

### Sample Sort CUDA
Graphs:
https://github.com/YidaZou/Project/blob/master/Analysis/sampleSortCUDA/sampleSortCUDA_Graphs.pdf

There is a significant increase in time across all number of threads as the input size increases.
The time seems to be holding constant over all number of threads for all input sizes. This may be a problem of my sorting algorithm not working correctly.
At least, in comp_large, the 2^12 input size seems to show somewhat of a downward trend in time as the number of threads increases.

### Sample Sort MPI

There is in increase in computational and communication time as the size to be sorted increases. It is expected that as the computational and comunicaiton time will decrease with an increased number of threads. It is expected that there are marginal differences between the differnt types of array population: random, sorted, reverse sorted, and 1% perturped. While one would expect a sorted array to have statistically significant reduction in time say compared to a randomly sorted aray, the overhead of communication time and the computational splitting make the differnce less prevelant. I was not able to collect all of the .cali files for every combination of parameters. Specifically, my jobs with 512 and 1024 number of threads either timed out or failed due to nodes being killed on Grace. I plan to provide graphs and every parameter combination of .cali files before the completion of the project.

### Bitonic Sort CUDA

Although graphs were not completed in time, we can expect an increase in the computational and communication time as the size of the array increase. We can also expect bitonic sort parallelism to reach a point of diminishing returns as we increase the number of threads. As we increase the number of processes, the algorithmic runtime should decrease.

### Bitonic Sort MPI

Although graphs were not completed in time, we can expect bitonic sort parallelism to reach a point of diminishing returns as we increase the number of processes. As we increase the number of processes, the algorithmic runtime should decrease.


### 4a. Vary the following parameters
For inputSizes:
- 2^16, 2^18, 2^20, 2^22, 2^24, 2^26, 2^28

For inputTypes:
- Sorted, Random, Reverse sorted, 1%perturbed

num_procs, num_threads:
- MPI: num_procs:
    - 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- CUDA: num_threads:
    - 64, 128, 256, 512, 1024

This should result in 4x7x10=280 Caliper files for your MPI experiments.

### 4b. Hints for performance analysis

To automate running a set of experiments, parameterize your program.

- inputType: If you are sorting, "Sorted" could generate a sorted input to pass into your algorithms
- algorithm: You can have a switch statement that calls the different algorithms and sets the Adiak variables accordingly
- num_procs:   How many MPI ranks you are using
- num_threads: Number of CUDA or OpenMP threads

When your program works with these parameters, you can write a shell script
that will run a for loop over the parameters above (e.g., on 64 processors,
perform runs that invoke algorithm2 for Sorted, ReverseSorted, and Random data).

### 4c. You should measure the following performance metrics
- `Time`
    - Min time/rank
    - Max time/rank
    - Avg time/rank
    - Total time
    - Variance time/rank
    - `If GPU`:
        - Avg GPU time/rank
        - Min GPU time/rank
        - Max GPU time/rank
        - Total GPU time

`Intel top-down`: For your CPU-only experiments on the scale of a single node, you should
generate additional performance data, measuring the hardware counters on the CPU. This can be done by adding `topdown.all` to the `spot()` options in the `CALI_CONFIG` in your jobfile.

## 5. Presentation

## 6. Final Report
Submit a zip named `TeamX.zip` where `X` is your team number. The zip should contain the following files:
- Algorithms: Directory of source code of your algorithms.
- Data: All `.cali` files used to generate the plots seperated by algorithm/implementation.
- Jupyter notebook: The Jupyter notebook(s) used to generate the plots for the report.
- Report.md
