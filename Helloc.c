#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_SIZE 10000000

void funcArray(double* arr, size_t len) {
	//double* temp = (double*)malloc(sizeof(double) * len);
	double step = 3.141592653589783 * 2 / MAX_SIZE;

	#pragma acc data copyin(step)
	#pragma acc parallel loop vector vector_length(128) gang, present_or_copyin(arr)
	for (size_t i = 0; i < len; i++)
	{
		arr[i] = sin(step * i);
	}
}

double sumArray(double* arr, size_t len) {
	double sum = 0.0;

	#pragma acc data copy(sum)
	#pragma acc parallel loop reduction(+:sum) present(arr)
	for (size_t i = 0; i < len; i++)
	{
		sum += arr[i];
	}

	return sum;
}

int main() {
	double sec = 0.0;

	double* arr = (double*)malloc(sizeof(double) * MAX_SIZE);
	#pragma acc data create(arr[0:MAX_SIZE])
	{
	clock_t before = clock();

	funcArray(arr, N);
	printf("summa = %0.23f\n", sumArray(arr, MAX_SIZE));

	sec += (double)(clock() - before)/ CLOCKS_PER_SEC;
	printf("Time taken: %.5f", sec);

	free(arr);
	}
	return 0;
}
