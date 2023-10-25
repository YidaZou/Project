# CSCE 435 Group project

## 1. Group members:
1. Yida Zou
2. Second
3. Third
4. Fourth

---

## 2. _due 10/25_ Project topic

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

For example:
- Algorithm 1a (MPI + CUDA)
- Algorithm 1b (MPI on each core)
- Algorithm 2a (MPI + CUDA)
- Algorithm 2b (MPI on each core)


We will communicate via iMessage.

Algorithms we will use:
- Sample Sort
```
function SampleSort(unsortedList, t):  //t = thread count
    //Divide input into samples depending on number of threads
    sampleSize = calculateSampleSize(unsortedList, t)
    sample = selectSample(unsortedList, sampleSize)
    
    //Distribute the sample to all processors 
    distribute(sample)  //Using MPI
    
    //Each thread sorts the given sample locally
    sortedSample = sortLocally(sample)

    //Gather sorted samples
    sortedSamples = communicate(sample)    //Using MPI

    //Merge and sort all samples together
    sortedLists = mergeSublists(sortedSamples)
    
    return sortedSublist
```
