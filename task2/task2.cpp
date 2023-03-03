#include <iostream>
#include <string>
#include <cmath>



#define CORN1 10
#define CORN2 20
#define CORN3 30
#define CORN4 20
#define ITER 10000000
#define ACC 0.000001
#define GRID_SIZE 1024

#define ABS(a) ((a < 0) ? (-a) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main() {
	double** newa = new double* [GRID_SIZE];
	for (size_t i = 0; i < GRID_SIZE; i++)
		newa[i] = new double[GRID_SIZE];

	double** olda= new double* [GRID_SIZE];
	for (size_t i = 0; i < GRID_SIZE; i++)
		olda[i] = new double[GRID_SIZE];

	clock_t beforeinit = clock();
	#pragma acc data copy (olda[0:GRID_SIZE][0:GRID_SIZE]), copy (newa[0:GRID_SIZE][0:GRID_SIZE]) {
	olda[0][0] = CORN1;
	olda[GRID_SIZE-1][0] = CORN2;
	olda[0][GRID_SIZE - 1] = CORN3;
	olda[GRID_SIZE - 1][GRID_SIZE - 1] = CORN4;
	double prop1 = (CORN2 - CORN1)/ (GRID_SIZE - 2);
	double prop2 = (CORN3 - CORN1) / (GRID_SIZE - 2);
	double prop3 = (CORN4 - CORN3) / (GRID_SIZE - 2);
	double prop4 = (CORN2 - CORN3) / (GRID_SIZE - 2);
	#pragma acc parallel loop seq gang num_gangs(256) vector vector_length(256)
	for (size_t i = 1; i < GRID_SIZE - 1; i++) {
		olda[0][i] = olda[0][i - 1] + prop1;
		olda[i][0] = olda[i - 1][0] + prop2;
		olda[GRID_SIZE - 1][i] = olda[GRID_SIZE - 1][i - 1] + prop3;
		olda[i][GRID_SIZE - 1] = olda[i - 1][GRID_SIZE - 1] + prop4;
		newa[0][i] = olda[0][i];
		newa[i][0] = olda[i][0];
		newa[GRID_SIZE - 1][i] = olda[GRID_SIZE - 1][i];
		newa[i][GRID_SIZE - 1] = olda[i][GRID_SIZE - 1];
	}

	std::cout << "Initialization time: " << 1.0 * (clock() - beforeinit) / CLOCKS_PER_SEC << std::endl;
	clock_t beforecal = clock();

	int iter_count = 0;
	double error = 0.0;
	#pragma acc parallel loop seq gang num_gangs(256) vector vector_length(256) 
	while (iter_count < ITER && error < ACC) {
		for (size_t i = 1; i < GRID_SIZE - 1; i++) {
			for (size_t j = 1; j < GRID_SIZE - 1; j++) {
				newa[i][j] = (olda[i + 1][j] + olda[i - 1][j] + olda[i][j - 1] + olda[i][j + 1]) / 4;
				error = MAX(error, ABS(newa[i][j] - olda[i][j]));
			}
		}
		iter_count++;
	}

	std::cout << "Iteration: " << iter_count << " " << "Error: " << error << std::endl;
	std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;
	}
	
	return 0
}