//
// Created by Poodar Chu on 4/20/17.
//

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <ctime>
#include <sys/time.h>

const int MAX_LEN=10000;

void Odd_even_sort(
        int  a[]  /* in/out */,
        int  n    /* in     */) {
    int phase, i, temp;

    for (phase = 0; phase < n; phase++)
        if (phase % 2 == 0) { /* Even phase */
            for (i = 1; i < n; i += 2)
                if (a[i-1] > a[i]) {
                    temp = a[i];
                    a[i] = a[i-1];
                    a[i-1] = temp;
                }
        } else { /* Odd phase */
            for (i = 1; i < n-1; i += 2)
                if (a[i] > a[i+1]) {
                    temp = a[i];
                    a[i] = a[i+1];
                    a[i+1] = temp;
                }
        }
}  /* Odd_even_sort sequential*/

void swap(int *x, int *y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}

int main(int argc, char* argv[]) {

    int arr[MAX_LEN];
    int value[MAX_LEN];

    int n_per_processor, minindex;

    MPI_Status status;

    srand(time(0));

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start, finish;
    double time_used;

    if(rank == 0) {

        printf("Enput num of numbers processed per processor?\n");
        scanf("%d", &n_per_processor);

        start = MPI_Wtime();

        printf("Num_Cpu: %d, # per cpu: %d\n", size, n_per_processor);

        if (size * n_per_processor < MAX_LEN) {
            printf("Initial array: \n");
            for (int i = 0; i < size*n_per_processor; ++i) {
                arr[i] = rand() % 10000;
                printf("%d ", arr[i]);
            }
            printf("\n");

            struct timeval start2, end;
            gettimeofday(&start2, nullptr);
            Odd_even_sort(arr, size*n_per_processor);
            gettimeofday(&end, nullptr);

            time_used = 1000000*(end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec);
            time_used /= 1000000.0;

        } else
            return -1;
    }



    MPI_Bcast(&n_per_processor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&arr, n_per_processor, MPI_INT, &value, n_per_processor, MPI_INT, 0, MPI_COMM_WORLD);

    for(int n = 0; n < size; ++n) {
        if( n % 2 == 0) {
            if( rank % 2 == 0) {

                MPI_Send(&value[0], n_per_processor, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
                MPI_Recv(&value[n_per_processor], n_per_processor, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &status);

                for(int i = 0; i < (n_per_processor*2 - 1); ++i) {
                    minindex = i;

                    for(int j = i + 1; j < n_per_processor*2; ++j)
                        if(value[j] < value[minindex])
                            minindex = j;

                    if(minindex > i)
                        swap(&value[i], &value[minindex]);
                }

            } else {
                MPI_Recv(&value[n_per_processor], n_per_processor, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&value[0], n_per_processor, MPI_INT, rank-1, 0, MPI_COMM_WORLD);

                for(int i = 0; i < (n_per_processor*2 - 1); ++i) {
                    minindex = i;

                    for(int j = i + 1; j < n_per_processor*2; ++j)
                        if(value[j] < value[minindex])
                            minindex = j;

                    if(minindex > i)
                        swap(&value[i], &value[minindex]);
                }

                for(int i=0;i<n_per_processor;i++)
                    swap(&value[i],&value[i+n_per_processor]);
            }
        }
        else {
            if(rank%2 == 1 && rank != (size-1)) {

                MPI_Send(&value[0], n_per_processor, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
                MPI_Recv(&value[n_per_processor], n_per_processor, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &status);

                for(int i = 0; i < (n_per_processor*2 - 1); ++i) {
                    minindex = i;

                    for(int j = i + 1; j < n_per_processor*2; ++j)
                        if(value[j] < value[minindex])
                            minindex = j;

                    if(minindex > i)
                        swap(&value[i], &value[minindex]);
                }
            } else if(rank != 0 && rank != (size-1)) {
                MPI_Recv(&value[n_per_processor], n_per_processor, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&value[0], 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);

                for(int i = 0; i < (n_per_processor*2 - 1); i++) {
                    minindex = i;

                    for(int j = i + 1; j < n_per_processor*2; j++)
                        if(value[j] < value[minindex])
                            minindex = j;

                    if(minindex > i)
                        swap(&value[i], &value[minindex]);
                }

                for(int i=0;i<n_per_processor;i++)
                    swap(&value[i],&value[i+n_per_processor]);
            }
        }
    }

    //MPI_Scatter(&arr,1,MPI_INT,&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&value[0], n_per_processor, MPI_INT, &arr[0], n_per_processor, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        printf("Sorted array: \n");

        for (int i = 0; i < size*n_per_processor; ++i)
            printf("%d ", arr[i]);

        printf("\n\n");

        finish = MPI_Wtime();
        printf("Parallel Elapsed time = %e seconds\n", finish-start);
        printf("Sequential Elapsed time = %e seconds\n\n", time_used);

        printf("Accelerate Ratio is %e\n", time_used/(finish-start));
    }

    MPI_Finalize();

    return 0;
}