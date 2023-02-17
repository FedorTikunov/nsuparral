#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define FORMAT float

#if(FORMAT == float)
	#define PFORMAT "summa = %0.23f\n"	
endif

#if(FORMAT == double)
	#define PFORMAT "summa = %0.23lf\n"	
endif

#define MAX_SIZE 10000000

void funcArray(FORMAT* arr, size_t len) {

	FORMAT step = 3.141592653589783 * 2 / MAX_SIZE;

	#pragma acc data copyin(step)
	#pragma acc parallel loop gang num_gangs(2048) vector vector_length(64), present(arr)
	for (size_t i = 0; i < len; i++)
	{
		arr[i] = sin(step * i);
	}
}

FORMAT sumArray(FORMAT* arr, size_t len) {
	FORMAT sum = 0.0;

	#pragma acc data copy(sum)
	#pragma acc parallel loop gang num_gangs(2048) reduction(+:sum) present(arr)
	for (size_t i = 0; i < len; i++)
	{
		sum += arr[i];
	}

	return sum;
}

int main() {
	double sec = 0.0;

	FORMAT* arr = (FORMAT*)malloc(sizeof(FORMAT) * MAX_SIZE);
	#pragma acc data create(arr[0:MAX_SIZE])
	{
	clock_t before = clock();

	funcArray(arr, MAX_SIZE);
	printf(PFORMAT, sumArray(arr, MAX_SIZE));
	sec += (FORMAT)(clock() - before)/ CLOCKS_PER_SEC;
	printf("Time taken: %.5f", sec);

	free(arr);
	}
	return 0;
}
