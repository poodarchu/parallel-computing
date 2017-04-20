//
// Created by Poodar Chu on 4/20/17.
//

#include "mpi.h"
#include <cstdio>
#include <sys/time.h>

#include <stdlib.h>

#define NUM_ROW_A 100                 /* number of rows in matrix A */
#define NUM_COL_A 150                /* number of columns in matrix A */
#define NUM_COL_B 200                  /* number of columns in matrix B */

#define FROM_MASTER_TAG 1          /* setting a message type */
#define FROM_WORKER_TAG 2          /* setting a message type */

int main (int argc, char *argv[]) {

    int	comm_size, my_rank, num_workers, source, dest, msg_type_tag, rows, ave_row, extra, offset;
    int i, j, k, rc;

    double	A[NUM_ROW_A][NUM_COL_A], B[NUM_COL_A][NUM_COL_B], C[NUM_ROW_A][NUM_COL_B];

    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_size < 2 ) {
        printf("Need at least two MPI tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    num_workers = comm_size-1;

    double p_start, p_end;
    double time_used;

    // Master rank
    if (my_rank == 0) {

        printf("Start with %d tasks.\n", comm_size);
        printf("Initializing arrays...\n");

        for (i=0; i<NUM_ROW_A; i++)
            for (j=0; j<NUM_COL_A; j++)
                A[i][j]= i+j;

        for (i=0; i<NUM_COL_A; i++)
            for (j=0; j<NUM_COL_B; j++)
                B[i][j]= i*j;

        struct timeval start2, end;
        gettimeofday(&start2, nullptr);
        double temp[NUM_ROW_A][NUM_COL_B];
        for (int i = 0; i < NUM_ROW_A; ++i)
            for (int j = 0; j < NUM_COL_B; ++j)
                for (int inner = 0; inner < NUM_COL_A; ++inner)
                    temp[i][j] += A[i][inner]*B[inner][j];
        gettimeofday(&end, nullptr);

        time_used = 1000000*(end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec);
        time_used /= 1000000.0;

        p_start = MPI_Wtime();

        /* Send matrix data to the worker tasks */
        ave_row = NUM_ROW_A / num_workers;
        extra = NUM_ROW_A % num_workers;
        offset = 0;
        msg_type_tag = FROM_MASTER_TAG;

        for (dest=1; dest <= num_workers; dest++) {
            rows = (dest <= extra) ? ave_row+1 : ave_row;

            printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);

            MPI_Send(&offset, 1, MPI_INT, dest, msg_type_tag, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, msg_type_tag, MPI_COMM_WORLD);
            MPI_Send(&A[offset][0], rows*NUM_COL_A, MPI_DOUBLE, dest, msg_type_tag, MPI_COMM_WORLD);
            MPI_Send(&B, NUM_COL_A*NUM_COL_B, MPI_DOUBLE, dest, msg_type_tag, MPI_COMM_WORLD);

            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        msg_type_tag = FROM_WORKER_TAG;
        for (i=1; i <= num_workers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, msg_type_tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, msg_type_tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset][0], rows*NUM_COL_B, MPI_DOUBLE, source, msg_type_tag, MPI_COMM_WORLD, &status);

            printf("Received results from task %d\n",source);
        }

        /* Print results */

        printf("Output Matrix:\n");
        for (i=0; i < NUM_ROW_A; i++) {
            printf("\n");
            for (j=0; j < NUM_COL_B; j++)
                printf("%6.2f   ", C[i][j]);
        }

        printf ("Finished.\n");
    }


    if (my_rank > 0) {
        msg_type_tag = FROM_MASTER_TAG;
        MPI_Recv(&offset, 1, MPI_INT, 0, msg_type_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, 0, msg_type_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&A, rows*NUM_COL_A, MPI_DOUBLE, 0, msg_type_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(&B, NUM_COL_A*NUM_COL_B, MPI_DOUBLE, 0, msg_type_tag, MPI_COMM_WORLD, &status);

        for (k=0; k<NUM_COL_B; k++)
            for (i=0; i<rows; i++)
            {
                C[i][k] = 0.0;
                for (j=0; j<NUM_COL_A; j++)
                    C[i][k] = C[i][k] + A[i][j] * B[j][k];
            }
        msg_type_tag = FROM_WORKER_TAG;
        MPI_Send(&offset, 1, MPI_INT, 0, msg_type_tag, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, msg_type_tag, MPI_COMM_WORLD);
        MPI_Send(&C, rows*NUM_COL_B, MPI_DOUBLE, 0, msg_type_tag, MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        p_end = MPI_Wtime();
        printf("\nSequential Running Time = %f seconds.\n", p_end-p_start);
        printf("Parallel Running Time = %f seconds.\n\n", time_used);

        printf("Accelerate Ratio = %e\n", (p_end-p_start)/time_used);
    }

    MPI_Finalize();
}