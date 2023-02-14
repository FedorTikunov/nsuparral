#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100000000

void funcArray(double* arr, size_t len) {
	//double* temp = (double*)malloc(sizeof(double) * len);
	double step = 2 * 3.14159265358979323846 / N;
	#pragma acc data copyin(step)
	#pragma acc parallel loop vector vector_length(128) gang
	for (size_t i = 0; i < len; i++) {
		arr[i] = sin(i * step);
	}
}

double sumArray(double* arr, size_t len) {
	double sum = 0.0;
	#pragma acc data copy(sum)
	#pragma acc parallel loop reduction(+:sum) present(arr)
	for (size_t  i = 0; i < len; i++) {
		sum += arr[i];
	}
	return sum;
}

int main() {
	double sec = 0.0

	double* arr = (double*)malloc(sizeof(double)*N);
	#pragma acc data create(arr[0:N])
	{
	clock_t before = clock();

	funcArray(arr, N);
	printf("summa = %0.23f\n", sumArray(arr, N));

	sec += (double)(clock() - before)/ CLOCKS_PER_SEC;
	printf("Time taken: %.5f", sec);

	free(arr);
	}
	return 0;
}
