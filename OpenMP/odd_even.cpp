#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

void odd_even_sort(int*a, int n);

int main(int argc, char* argv[]) {
	int a[] = { 1, 23, 14, 32, 7, 9, 13, 17, 25 };
	int n = 9;

	odd_even_sort(a, n);

	for (int i = 0; i < n; i++) {
		std::cout << a[i] << std::endl;
	}
	return 0;
}

void odd_even_sort(int*a, int n) {
	int phase, i, tmp;
	int thread_count = 8;
# pragma omp parallel num_threads(thread_count) default(none) shared(a, n) private(i, tmp, phase)
	for (phase = 0; phase < n; phase++) {
		if (phase % 2 == 0)
#     pragma omp for    
		for (i = 1; i < n; i += 2) {
			if (a[i - 1] > a[i]) {
				tmp = a[i - 1];
				a[i - 1] = a[i];
				a[i] = tmp;
			}
		}
		else
#         pragma omp for
		for (i = 1; i < n - 1; i += 2) {
			if (a[i] > a[i + 1]) {
				tmp = a[i + 1];
				a[i + 1] = a[i];
				a[i] = tmp;
			}
		}
	}
}
