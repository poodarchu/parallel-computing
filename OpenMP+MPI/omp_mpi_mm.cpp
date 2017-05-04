#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include<omp.h>
#include<math.h>

/* Main Program */

int main(int argc, char **argv)
{

    int             Numprocs, MyRank, iam;
    int             NoofCols, NoofRows, VectorSize, ScatterSize;
    int             index, irow, icol, iproc;
    int             Root = 0, ValidOutput = 1;
    float         **Matrix, *Buffer, *Mybuffer, *Vector, *MyFinalVector, *FinalVector;
    float          *CheckResultVector;
    FILE           *fp;
    int             MatrixFileStatus = 1, VectorFileStatus = 1;


    /* ........MPI Initialisation ....... */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
    MPI_Comm_size(MPI_COMM_WORLD, &Numprocs);

    if (MyRank == 0) {
        /* .......Read The Input File ...... */
        if ((fp = fopen("/Users/poodar/Developments/Cpp/CCLProjects/OpenMP-MPI/data/mdata.inp", "r")) == NULL) {
            MatrixFileStatus = 0;
        }
        if (MatrixFileStatus != 0) {
            fscanf(fp, "%d %d\n", &NoofRows, &NoofCols);

            /*
             * ...Allocate Memory And Read Matrix From File
             * .......
             */
            Matrix = (float **) malloc(NoofRows * sizeof(float *));
            for (irow = 0; irow < NoofRows; irow++) {
                Matrix[irow] = (float *) malloc(NoofCols * sizeof(float));
                for (icol = 0; icol < NoofCols; icol++) {
                    fscanf(fp, "%f", &Matrix[irow][icol]);
                }
            }
            fclose(fp);

            /* .......Convert 2-D Matrix Into 1-D Array ..... */
            Buffer = (float *) malloc(NoofRows * NoofCols * sizeof(float));

            index = 0;
            for (irow = 0; irow < NoofRows; irow++) {
                for (icol = 0; icol < NoofCols; icol++) {
                    Buffer[index] = Matrix[irow][icol];
                    index++;
                }
            }
        }

        /* Read Vector From Input File */
        if ((fp = fopen("/Users/poodar/Developments/Cpp/CCLProjects/OpenMP-MPI/data/vdata.inp", "r")) == NULL) {
            VectorFileStatus = 0;
        }
        if (VectorFileStatus != 0) {
            fscanf(fp, "%d\n", &VectorSize);

            Vector = (float *) malloc(VectorSize * sizeof(float));
            for (index = 0; index < VectorSize; index++)
                fscanf(fp, "%f", &Vector[index]);

        }
    } /* End  Of If Myrank = 0 */


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&MatrixFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
    if (MatrixFileStatus == 0) {
        if (MyRank == Root)
            printf("Can't Open Input File For Matrix ..... \n");
        MPI_Finalize();
        exit(-1);
    }

    MPI_Bcast(&VectorFileStatus, 1, MPI_INT, Root, MPI_COMM_WORLD);
    if (VectorFileStatus == 0) {
        if (MyRank == Root)
            printf("Can't Open Input File For Vector ..... \n");
        MPI_Finalize();
        exit(-1);
    }

    MPI_Bcast(&NoofRows, 1, MPI_INT, Root, MPI_COMM_WORLD);

    if (NoofRows < Numprocs) {
        if (MyRank == 0)
            printf("No Of Rows Should Be More Than No Of Processors ... \n");
        MPI_Finalize();
        exit(0);
    }

    if (NoofRows % Numprocs != 0) {
        if (MyRank == 0)
            printf("Matrix Cannot Be Striped Evenly ..... \n");
        MPI_Finalize();
        exit(0);
    }

    MPI_Bcast(&NoofCols, 1, MPI_INT, Root, MPI_COMM_WORLD);
    MPI_Bcast(&VectorSize, 1, MPI_INT, Root, MPI_COMM_WORLD);

    if (VectorSize != NoofCols) {
        if (MyRank == 0) {
            printf("Invalid Input Data..... \n");
            printf("NoofCols Should Be Equal To VectorSize\n");
        }
        MPI_Finalize();
        exit(0);
    }

    if (MyRank != 0)
        Vector = (float *) malloc(VectorSize * sizeof(float));
    MPI_Bcast(Vector, VectorSize, MPI_FLOAT, Root, MPI_COMM_WORLD);

    ScatterSize = NoofRows / Numprocs;
    Mybuffer = (float *) malloc(ScatterSize * NoofCols * sizeof(float));
    MPI_Scatter(Buffer, ScatterSize * NoofCols, MPI_FLOAT, Mybuffer,
                ScatterSize * NoofCols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MyFinalVector = (float *) malloc(ScatterSize * sizeof(float));

    for (irow = 0; irow < ScatterSize; irow++)
        MyFinalVector[irow] = 0;

    printf("\n");


    /* OpenMP Parallel Directive */

/*
#pragma omp parallel  private(iam)
	{

	 OpenMP Parallel For Directive */

    omp_set_num_threads(4);
#pragma omp parallel for private(index,icol,iam)
    for (irow = 0; irow < ScatterSize; irow++) {
        printf("The Threadid is %d with  each processor Rank %d\n", omp_get_thread_num(), MyRank);
        MyFinalVector[irow] = 0;
        index = irow * NoofCols;
        for (icol = 0; icol < NoofCols; icol++)
            MyFinalVector[irow] += (Mybuffer[index++] * Vector[icol]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MyRank == 0)
        FinalVector = (float *) malloc(NoofRows * sizeof(float));


    MPI_Gather(MyFinalVector, ScatterSize, MPI_FLOAT, FinalVector,
               ScatterSize, MPI_FLOAT, Root, MPI_COMM_WORLD);

    if (MyRank == 0) {
        printf("\n");
        printf(" --------------------------------------------------- \n");
        /* printf("Results of Gathering Data  %d: \n", MyRank); */
        printf("\n");

        for (index = 0; index < NoofRows; index++)
            printf(" FinalVector[%d] = %f \n", index, FinalVector[index]);
        printf(" --------------------------------------------------- \n");
    }
    if (MyRank == 0) {
        CheckResultVector = (float *) malloc(NoofRows * sizeof(float));
        for (irow = 0; irow < NoofRows; irow++) {
            CheckResultVector[irow] = 0;
            for (icol = 0; icol < NoofCols; icol++) {
                CheckResultVector[irow] += (Matrix[irow][icol] * Vector[icol]);
            }
            if (fabs((double) (FinalVector[irow] - CheckResultVector[irow])) >
                1.0E-10) {
                printf("Error %d\n", irow);
                ValidOutput = 0;
            }
        }
        if (ValidOutput)
            printf("\n-------Correct Result------\n");
        /* Freeing Allocated Memory */
        free(Matrix);
        free(Vector);
        free(Buffer);
        free(FinalVector);
        free(CheckResultVector);
    }

    /* Freeing Allocated Memory */
    free(Mybuffer);
    free(MyFinalVector);

    /* MPI-Termination */

    MPI_Finalize();

    return 0;
}

