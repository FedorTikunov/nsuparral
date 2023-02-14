#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 100000000
tempV = 2 * 3.14159265358979323846 / N;
void FuncArray(double** my_array, int len) {
	double* temp = (double*)malloc(sizeof(double) * len);
#pragma acc kernels
	{
		for (int i = 0; i < len; i++)
			temp[i] = sin(i * tempV);
		*my_array = temp;
	}
}

double SumArray(double** my_array, int len) {
	double sum = 0;
#pragma acc kernels
	{
		for (int i = 0; i < len; i++) {
			sum += *my_array[i];
		}
	}
	return sum;
}

int main() {
	clock_t before = clock();
	double** array = (double**)malloc(sizeof(double*));
	long long len = N;
#pragma acc data create(array[:N])
	{
		FuncArray(array, len);
		printf("summa = %.10f", SumArray(array, len));
	}
	clock_t difference = clock() - before;
	double msec = difference * 1000.0 / CLOCKS_PER_SEC;
	printf("Time taken: %.5f", msec/1000);
	return 0;
}
