#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

void bitonicMerge(int* a, int l, int r, int dir);
void bitonicSort(int* a, int l, int cnt, int dir);


void bitonicMerge(int* a, int l, int r, int dir) {
    if (r > 1) {
        int k = r / 2;
        for (int i = l; i < l + k; i++) {
            if ((a[i] > a[i + k]) == dir) {
                // swap
                int temp = a[i];
                a[i] = a[i + k];
                a[i + k] = temp;
            }
        }
        bitonicMerge(a, l, k, dir);
        bitonicMerge(a, l + k, k, dir);
    }
}

void bitonicSort(int* a, int l, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(a, l, k, 1);
        bitonicSort(a, l + k, k, 0);
        bitonicMerge(a, l, cnt, dir);
    }
}
