#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 100000000

void FuncArray(double** my_array, int len) {
	double* temp = (double*)malloc(sizeof(double) * len);
#pragma acc kernels
	{
	for (int i = 0; i < len; i++)
		temp[i] = sin(i);
	*my_array = temp;
	}
	
}


int main() {
	clock_t before = clock();
	double** array = (double**)malloc(sizeof(double*));
	long long len = N;
	FuncArray(array, len);
	clock_t difference = clock() - before;
	int msec = difference * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d", msec/1000);
	return 0;
}
