# CSCE 435 Group project

## 1. Group members:
1. Yida Zou
2. Second
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
