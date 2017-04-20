#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <iostream>

void Trap(double a, double b, int n, double* global_result);
double f(double x);

int main() {
	double global_result = 0.0;
	double a = 1.0, b = 10.0;
	int n = 10000;
	int thread_count = 8;

//	thread_count = strtol(argv[1], nullptr, 10);
//	printf("Enter a, b, and n\n");
//	scanf("%1f %1f %d", &a, &b, &n);
# pragma omp parallel num_threads(thread_count)
	Trap(a, b, n, &global_result);

	printf("With n = %d trapezoids, our estimate\n", n);
	printf("of the intefral from %f to %f = %.14e\n", a, b, global_result);

	return 0;
}

void Trap(double a, double b, int n, double* global_result_p) {
	double h, x, my_result;
	double local_a, local_b;
	int i, local_n;
	int my_rank = omp_get_thread_num();
	int thread_count = omp_get_num_threads();

	h = (b - a) / n;
	local_n = n / thread_count;
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;

	my_result = (f(local_a) + f(local_b)) / 2.0;
	for (i = 0; i < local_n - 1; i++) {
		x = local_a + i*h;
		my_result += f(x);
	}
	my_result = my_result*h;

# pragma omp critical
	*global_result_p += my_result;
}

double f(double x){
	return x*x;
}
