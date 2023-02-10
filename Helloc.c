#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 100000000

void FuncArray(double** my_array, int len) {
#pragma acc parallel{
	double* temp = (double*)malloc(sizeof(double) * len);
	for (int i = 0; i < len; i++)
		temp[i] = sin(i);
	*my_array = temp;
	}
}


int main() {
	double** array = (double**)malloc(sizeof(double**));
	long long len = N;
	FuncArray(array, len);
}
