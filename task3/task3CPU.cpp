#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define CORN1 10.0
#define CORN2 20.0
#define CORN3 30.0
#define CORN4 20.0

int main(int argc, char** argv)
{

	const int GRID_SIZE = std::stoi(argv[2]);
	const double ACC = std::pow(10, -(std::stoi(argv[1])));
	const int ITER = std::stoi(argv[3]);

	double* newa = new double[GRID_SIZE * GRID_SIZE];
	double* olda = new double[GRID_SIZE * GRID_SIZE];

	double* c_newa;
	double* c_olda;

	std::memset(olda, 0, GRID_SIZE * GRID_SIZE * sizeof(double));

	cublasStatus_t status;
	cublasHandle_t handler;
	cudaError err;

	status = cublasCreate(&handler);


	int iter_count = 0;
	double error = 1.0;
	int index = 0;

	olda[0] = CORN1;
	double prop1 = (CORN2 - CORN1) / (GRID_SIZE);
	double prop2 = (CORN3 - CORN1) / (GRID_SIZE);
	double prop3 = (CORN4 - CORN3) / (GRID_SIZE);
	double prop4 = (CORN2 - CORN4) / (GRID_SIZE);

	olda[0] = CORN1;
	olda[(GRID_SIZE - 1) * GRID_SIZE] = CORN3;
	olda[GRID_SIZE - 1] = CORN2;
	olda[GRID_SIZE - 1 + GRID_SIZE * (GRID_SIZE - 1)] = CORN4;
	newa[0] = CORN1;
	newa[(GRID_SIZE - 1) * GRID_SIZE] = CORN3;
	newa[GRID_SIZE - 1] = CORN2;
	newa[GRID_SIZE - 1 + GRID_SIZE * (GRID_SIZE - 1)] = CORN4;

	cudaMalloc((void**)&c_newa, sizeof(double) * GRID_SIZE * GRID_SIZE);
	cudaMalloc((void**)&c_olda, sizeof(double) * GRID_SIZE * GRID_SIZE);

#pragma acc enter data copyin (olda[0:(GRID_SIZE * GRID_SIZE)], newa[0:(GRID_SIZE * GRID_SIZE)])
	{

		clock_t beforeinit = clock();
		double beta = -1.0;
		int index = 0;

#pragma acc data present(olda, newa)
#pragma acc parallel loop gang num_gangs(256) vector vector_length(256) async(1)
		for (size_t i = 1; i < GRID_SIZE - 1; i++) {
			olda[i] = olda[0] + prop1 * i;
			olda[i * GRID_SIZE] = olda[0] + prop2 * i;
			olda[(GRID_SIZE - 1) * GRID_SIZE + i] = olda[(GRID_SIZE - 1) * GRID_SIZE] + prop3 * i;
			olda[GRID_SIZE * i + GRID_SIZE - 1] = olda[GRID_SIZE * (GRID_SIZE - 1) + GRID_SIZE - 1] + prop4 * i;
			newa[i] = olda[i];
			newa[i * GRID_SIZE] = olda[i * GRID_SIZE];
			newa[(GRID_SIZE - 1) * GRID_SIZE + i] = olda[(GRID_SIZE - 1) * GRID_SIZE + i];
			newa[GRID_SIZE * i + GRID_SIZE - 1] = olda[GRID_SIZE * i + GRID_SIZE - 1];
		}
		std::cout << "Initialization time: " << 1.0 * (clock() - beforeinit) / CLOCKS_PER_SEC << std::endl;
		clock_t beforecal = clock();
#pragma acc wait(1) async(2)
		while (iter_count < ITER && error > ACC)
		{
			iter_count++;

#pragma acc data present(olda, newa)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) async(2)
			for (size_t i = 1; i < GRID_SIZE - 1; i++)
			{
				for (size_t j = 1; j < GRID_SIZE - 1; j++)
				{
					newa[i * GRID_SIZE + j] = 0.25 *
						(olda[(i + 1) * GRID_SIZE + j] +
							olda[(i - 1) * GRID_SIZE + j] +
							olda[i * GRID_SIZE + j - 1] +
							olda[i * GRID_SIZE + j + 1]);
				}
			}
			if (iter_count % 100 == 0)
			{
				cudaMemcpy(c_olda, olda, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
				cudaMemcpy(c_newa, newa, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
				status = cublasDaxpy(handler, GRID_SIZE * GRID_SIZE, &beta, newa, 1, c_olda, 1);
				status = cublasIdamax(handler, GRID_SIZE * GRID_SIZE, olda, 1, &index);
				cudaMemcpy(&error, &olda[index - 1], sizeof(double), cudaMemcpyDeviceToHost);
				error = std::abs(error);

				//status = cublasDcopy(handler, GRID_SIZE * GRID_SIZE, newa, 1, olda, 1);
			}

			double* temp = olda;
			olda = newa;
			newa = temp;
		}
		std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;
	}

	cudaFree(c_newa);
	cudaFree(c_olda);
	//std::cout << "Iteration: " << iter_count << " " << "Error : " << error << std::endl;
	printf("Last iteration: %d Error: %.6lf\n", iter_count, error);
	delete[] olda;
	delete[] newa;

	return 0;
}
