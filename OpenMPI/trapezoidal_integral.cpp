//
// Created by Poodar Chu on 4/20/17.
//

#include "mpi.h"
#include <stdio.h>
#include <sys/time.h>

double f(double x) {
    return x*x;
}


double Trap(double a, double b, int n, double h) {
    double integral;
    int k;

    integral = (f(a) + f(b))/2.0;
    for (k = 1; k <= n-1; k++) {
        integral += f(a+k*h);
    }
    integral = integral*h;

    return integral;
}  /* Trap sequential*/

double trap(double left_endpt,
            double right_endpt,
            int trap_count,
            double base_len) {
    double estimate, x;
    estimate = (f(left_endpt)+f(right_endpt))/2.0;

    for (int i = 0; i < trap_count-1; ++i) {
        x = left_endpt + i*base_len;
        estimate += f(x);
    }

    estimate = estimate * base_len;

    return estimate;
}

int main(int argc, char* argv[]) {

    double start, finish;
    double time_used;

    int my_rank, comm_size, n, local_n;
    double a , b, h, local_a, local_b;
    double local_int, total_int;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0) {
        printf("Enter a, b, and n\n");
        scanf("%lf %lf %d", &a, &b, &n);

        struct timeval start2, end;
        gettimeofday(&start2, nullptr);
        double integral = Trap(a, b, n, (b-a)/n);
        gettimeofday(&end, nullptr);

        time_used = 1000000*(start2.tv_sec-end.tv_sec) + (end.tv_usec-start2.tv_usec);
        time_used /= 1000000.0;

        start = MPI_Wtime();
    }

    MPI_Bcast(&a,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&b,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(&n,1,MPI_DOUBLE,0,MPI_COMM_WORLD);


    h = (b-a)/n;
    local_n = n/comm_size;

    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;
    local_int = trap(local_a, local_b, local_n, h);

    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        finish = MPI_Wtime();

        printf("With n = %d trapezoids, our estimate\n", n);
        printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
        printf("Parallel Elapsed time = %e seconds\n", finish-start);
        printf("Sequential Elapsed time = %e seconds\n", time_used);

        printf("\n Accelerate Ratio = %e\n", time_used/(finish-start));
    }

    MPI_Finalize();

    return 0;
}