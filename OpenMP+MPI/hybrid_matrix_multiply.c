#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int num_p, rank_id, iam;
    int num_cols, num_rows, vec_sz, scatter_sz;
    int index, irow, icol, iproc;

    int root = 0, valid_output_tag = 1;

    float **Matrix, *Buffer, *Mybuffer, *Vector, *MyFinalVector, *FinalVector;
    float *CheckResultVector;
    FILE *fp;

    int mat_status = 1, vec_status = 1;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    if (rank_id == 0) {
        if ((fp = fopen("/Users/poodar/Developments/Projects/parallel_computing/OpenMP+MPI/data/mdata", "r")) == NULL) {
            mat_status = 0;
        }
        if (mat_status != 0) {
            fscanf(fp, "%d %d\n", &num_rows, &num_cols);

            Matrix = (float **) malloc(num_rows * sizeof(float *));
            for (irow = 0; irow < num_rows; irow++) {
                Matrix[irow] = (float *) malloc(num_cols * sizeof(float));
                for (icol = 0; icol < num_cols; icol++) {
                    fscanf(fp, "%f", &Matrix[irow][icol]);
                }
            }
            fclose(fp);

            // Convert 2-D Matrix Into 1-D Array
            Buffer = (float *) malloc(num_rows * num_cols * sizeof(float));

            index = 0;
            for (irow = 0; irow < num_rows; irow++) {
                for (icol = 0; icol < num_cols; icol++) {
                    Buffer[index] = Matrix[irow][icol];
                    index++;
                }
            }
        }

        if ((fp = fopen("/Users/poodar/Developments/Projects/parallel_computing/OpenMP+MPI/data/vdata", "r")) == NULL) {
            vec_status = 0;
        }

        if (vec_status != 0) {
            fscanf(fp, "%d\n", &vec_sz);

            Vector = (float *) malloc(vec_sz * sizeof(float));
            for (index = 0; index < vec_sz; index++)
                fscanf(fp, "%f", &Vector[index]);

        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&num_rows, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&vec_sz, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (rank_id != 0)
        Vector = (float *) malloc(vec_sz * sizeof(float));

    MPI_Bcast(Vector, vec_sz, MPI_FLOAT, root, MPI_COMM_WORLD);

    scatter_sz = num_rows / num_p;
    Mybuffer = (float *) malloc(scatter_sz * num_cols * sizeof(float));

    MPI_Scatter(Buffer, scatter_sz * num_cols, MPI_FLOAT, Mybuffer, scatter_sz * num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MyFinalVector = (float *) malloc(scatter_sz * sizeof(float));

    for (irow = 0; irow < scatter_sz; irow++)
        MyFinalVector[irow] = 0;

    omp_set_num_threads(8);
#pragma omp parallel for private(index,icol,iam)
    for (irow = 0; irow < scatter_sz; irow++) {
        printf("Thread id: %d with processor rank: %d\n", omp_get_thread_num(), rank_id);
        MyFinalVector[irow] = 0;
        index = irow * num_cols;
        for (icol = 0; icol < num_cols; icol++)
            MyFinalVector[irow] += (Mybuffer[index++] * Vector[icol]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank_id == 0)
        FinalVector = (float *) malloc(num_rows * sizeof(float));


    MPI_Gather(MyFinalVector, scatter_sz, MPI_FLOAT, FinalVector,
               scatter_sz, MPI_FLOAT, root, MPI_COMM_WORLD);

    if (rank_id == 0) {
        for (index = 0; index < num_rows; index++)
            printf(" output[%d] = %f \n", index, FinalVector[index]);

        CheckResultVector = (float *) malloc(num_rows * sizeof(float));
        for (irow = 0; irow < num_rows; irow++) {
            CheckResultVector[irow] = 0;
            for (icol = 0; icol < num_cols; icol++) {
                CheckResultVector[irow] += (Matrix[irow][icol] * Vector[icol]);
            }
            if (fabs((double) (FinalVector[irow] - CheckResultVector[irow])) >
                1.0E-10) {
                printf("Error %d\n", irow);
                valid_output_tag = 0;
            }
        }
        if (valid_output_tag)
            printf("Result is correct.\n");

        free(Matrix);
        free(Vector);
        free(Buffer);
        free(FinalVector);
        free(CheckResultVector);
    }

    free(Mybuffer);
    free(MyFinalVector);

    MPI_Finalize();

    return 0;
}
