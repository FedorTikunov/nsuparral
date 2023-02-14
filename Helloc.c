#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 100000000
void FuncArray(double* my_array, size_t len) {
	//double* temp = (double*)malloc(sizeof(double) * len);
	double step = 2 * 3.14159265358979323846 / N;
	#pragma acc data copyin(step)
	#pragma acc parallel loop vector vector_length(128) gang
	for (size_t i = 0; i < len; i++) {
		my_array[i] = sin(i * step);
	}
}

double SumArray(double* my_array, size_t len) {
	double sum = 0.0;
	#pragma acc data copy(sum)
	#pragma acc parallel loop reduction(+:sum) present(my_array)
	for (size_t  i = 0; i < len; i++) {
		sum += my_array[i];
	}
	return sum;
}

int main() {
	double* array = (double*)malloc(sizeof(double)*N);
	#pragma acc data create(array[0:N])
	{
	clock_t before = clock();

	FuncArray(array, N);
	printf("summa = %0.23f\n", SumArray(array, N));

	double sec = (double)(clock() - before)/ CLOCKS_PER_SEC;
	printf("Time taken: %.5f", sec);

	free(array);
	}
	return 0;
}
