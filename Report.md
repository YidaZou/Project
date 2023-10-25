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
'''
function SampleSort(inputList, t):  //t = thread count
    # Step 1: Divide list into samples depending on number of threads
    sampleSize = calculateSampleSize(inputList)
    sample = selectSample(inputList, sampleSize)
    
    # Step 2: Broadcast Sample
    broadcast(sample)  # Distribute the sample to all processors
    
    # Step 3: Local Sort
    localList = sortLocally(inputList)  # Each processor sorts its local list
    
    # Step 4: Local Split
    localSublists = splitLocally(localList, sample)
    
    # Step 5: Global Merge
    sortedSublists = mergeSublists(localSublists)  # Merge all sorted sublists
    
    return sortedSublists
'''
