#include <iostream>
#include <string>
#include <cmath>
#include <ctime>


#define CORN1 10.0
#define CORN2 20.0
#define CORN3 30.0
#define CORN4 20.0
#define ITER 10000000
#define ACC 0.000001
#define GRID_SIZE 128

int main() {
	double* newa = new double[GRID_SIZE * GRID_SIZE];
	double* olda = new double[GRID_SIZE * GRID_SIZE];
	int iter_count = 0;
        double error = 1.0;
        olda[0] = CORN1;
	olda[(GRID_SIZE - 1) * GRID_SIZE] = CORN3;
	olda[GRID_SIZE - 1] = CORN2;
	olda[GRID_SIZE - 1 + GRID_SIZE * (GRID_SIZE - 1)] = CORN4;
	newa[0] = CORN1;
	newa[(GRID_SIZE - 1) * GRID_SIZE] = CORN3;
	newa[GRID_SIZE - 1] = CORN2;
	newa[GRID_SIZE - 1 + GRID_SIZE * (GRID_SIZE - 1)] = CORN4;
	double prop1 = (CORN2 - CORN1) / (GRID_SIZE);
	double prop2 = (CORN3 - CORN1) / (GRID_SIZE);
	double prop3 = (CORN4 - CORN3) / (GRID_SIZE);
	double prop4 = (CORN2 - CORN4) / (GRID_SIZE);
	for (size_t i = 1; i < GRID_SIZE - 1; i++) {
		olda[i] = olda[i - 1] + prop1;
		olda[i * GRID_SIZE] = olda[(i - 1) * GRID_SIZE] + prop2;
		olda[(GRID_SIZE - 1) * GRID_SIZE + i] = olda[(GRID_SIZE - 1) * GRID_SIZE + (i - 1)] + prop3;
		olda[GRID_SIZE * i + GRID_SIZE - 1] = olda[GRID_SIZE * (i - 1) + GRID_SIZE - 1] + prop4;
		newa[i] = olda[i];
		newa[i * GRID_SIZE] = olda[i * GRID_SIZE];
		newa[(GRID_SIZE - 1) * GRID_SIZE + i] = olda[(GRID_SIZE - 1) * GRID_SIZE + i];
		newa[GRID_SIZE * i + GRID_SIZE - 1] = olda[GRID_SIZE * i + GRID_SIZE - 1];
	 }
	#pragma acc enter data copyin(error, olda[0:(GRID_SIZE * GRID_SIZE)], newa[0:(GRID_SIZE * GRID_SIZE)])
	{	
	while (iter_count < ITER && error > ACC) {
	
	#pragma acc kernels async(1)
	error = 0.000001;
	#pragma acc update device(error) async(1)

	#pragma acc data present(newa, olda, error)
	#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max:error) async(1)
	for (size_t i = 1; i < GRID_SIZE - 1; i++) {
		for (size_t j = 1; j < GRID_SIZE - 1; j++) {
			newa[i * GRID_SIZE + j] = 0.25 * (olda[(i + 1) * GRID_SIZE + j] + olda[(i - 1) * GRID_SIZE + j] + olda[i * GRID_SIZE + j - 1] + olda[i * GRID_SIZE + j + 1]);
			error = fmax(error, fabs(newa[i * GRID_SIZE + j] - olda[i * GRID_SIZE + j]));
		}
	}
	#pragma acc update host(error) async(1)

	#pragma acc wait(1)
	iter_count++;
	double* c = olda;
	olda = newa;
	newa = c;
	}
	}
	std::cout << "Iteration: " << iter_count << " " << "Error: " << error << std::endl;
	return 0;
}
