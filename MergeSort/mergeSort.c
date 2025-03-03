// C program for Merge Sort
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
 
// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
 
    // Create temp arrays
    int L[n1], R[n2];
 
    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];
 
    // Merge the temp arrays back into arr[l..r
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    // Copy the remaining elements of L[],
    // if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    // Copy the remaining elements of R[],
    // if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}
 
// l is for left index and r is right index of the
// sub-array of arr to be sorted
void mergeSort(int arr[], int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
 
        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
 
        merge(arr, l, m, r);
    }
}
 
// Function to print an array
void printArray(int A[], int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}


// Check if sorted
void correctness_check(int arr[], int size) 
{
    int i;
    for(i = 0; i < size - 1; i++){
        if(arr[i] > arr[i+1]){
            printf("\nError: Array not sorted\n");
            return;
        }
    }
    printf("\nArray sorted correctly\n");
    return;
}

void fillRandomArray(int arr[], int size) {
    srand(time(NULL));  // Seed the random number generator with the current time

    for (int i = 0; i < size; i++) {
        arr[i] = rand();  // Generate random integers for the array
    }
}
 
// Driver code
int main()
{
    // Define an array with 1000 elements
  int arr_size = 10;
  int arr[arr_size]; 

  // Fill the array with random values
  fillRandomArray(arr, arr_size);
 
    printf("Given array is \n");
    printArray(arr, arr_size);
 
    mergeSort(arr, 0, arr_size - 1);
 
    printf("\nSorted array is \n");
    printArray(arr, arr_size);

    correctness_check(arr, arr_size);
    return 0;
}

