#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 100

int compare(const void *, const void *);
void mergeArrays(int *, int *, int *, int, int);
int computeNeighbor(int, int, int);

int main(int argc, char ** argv)
{
//    omp_set_num_threads(8);

    int i, j, n, rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numElements = N/size;
    int * arr = (int *) malloc(sizeof(int)*numElements);
    int * temp = (int *) malloc(sizeof(int)*numElements*2);
    int * recvArr = (int *) malloc(sizeof(int)*numElements);
    int * fullArr = NULL;

    if(rank == 0)
        fullArr = (int *) malloc(sizeof(int)*N);

    //Fill array in descending order
    int start = rank*numElements;
    int end = start+numElements;
    for(i = 0, j = start; i < numElements; i++, j++)
        arr[i] = N-j;

    //Sort array with odd-even sort

    //Sort local values
    qsort(arr, numElements, sizeof(int), compare);

    //Begin iterations
    for(n = 0; n < size; n++) {
        MPI_Barrier(MPI_COMM_WORLD);
        int neighbor = computeNeighbor(n, rank, size);

        if(neighbor >= 0 && neighbor < size)
        {
            //Send my values to my neighbor and receive values from my neighbor
            MPI_Sendrecv(arr, numElements, MPI_INT, neighbor, n,
                         recvArr, numElements, MPI_INT, neighbor, n,
                         MPI_COMM_WORLD, &status);

            //If my rank < my neighbor's rank, keep the smaller values
            if(rank < neighbor){
                mergeArrays(arr, recvArr, temp, numElements, 1);
                //Else keep the larger values
            } else {
                mergeArrays(arr, recvArr, temp, numElements, 0);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(arr, numElements, MPI_INT, fullArr, numElements, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        for(i = 0; i < N; i++)
            printf("%d ", fullArr[i]);
        printf("\n");
    }

    free((void *) arr);
    free((void *) recvArr);
    free((void *) temp);

    MPI_Finalize();
    return 0;
}

int compare(const void * a, const void * b)
{
    if( *((int *)a) <  *((int *)b) ) return -1;
    else if( *((int *)a) == *((int *)b) ) return 0;
    else return 1;
}

int computeNeighbor(int phase, int rank, int size)
{
    int neighbor;
    if(phase % 2 != 0) {  //odd phase
        if(rank % 2 != 0) {  //odd rank
            neighbor = rank + 1;
        } else {  //even rank
            neighbor = rank - 1;
        }
    } else {  //even phase
        if(rank % 2 != 0) {  //odd rank
            neighbor = rank - 1;
        } else {  //even rank
            neighbor = rank + 1;
        }
    }

    if(neighbor < 0 || neighbor >= size)
        neighbor = -1;
    return neighbor;
}

void mergeArrays(int * arr, int * neighborArr, int * temp, int size, int low_or_high)
{
    int i, j, k;
    i = j = k = 0;

    //Assume arrays are already sorted
//    j = 0, k = 0;
//#pragma omp parallel for private(j, k)
    for(i = 0, j = 0, k = 0; i < size*2; i++)
    {
        if(j < size && k < size)
        {
            if(arr[j] < neighborArr[k])
            {
                temp[i] = arr[j];
                j++;
            } else {
                temp[i] = neighborArr[k];
                k++;
            }
        } else if(j < size) {
            temp[i] = arr[j];
            j++;
        } else {
            temp[i] = neighborArr[k];
            k++;
        }
    }

    if(low_or_high % 2 != 0)
#pragma omp parallel for default(shared)
        for(i = 0; i < size; i++)
            arr[i] = temp[i];
    else {
        i = size;
//#pragma omp parallel for default(shared)
        for (j = 0; j < size;j++) {
            arr[j] = temp[i];
//#pragma omp critical
            i++;
        }
    }
}