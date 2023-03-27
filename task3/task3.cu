#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cublas_v2.h"

#define CORN1 10.0
#define CORN2 20.0
#define CORN3 30.0
#define CORN4 20.0

__global__ void MatAdd(double olda[][128*128], double newa[][128*128]) {
	        int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if  (i < 128 && j < 128) {
			olda[j* 128 + i ] = std::abs(olda[j * 128 + i] - newa[j * 128 + i]);
		}
}

int main(int argc, char** argv) {
	int GRID_SIZE = std::stoi(argv[2]);
	double ACC = std::pow(10, -(std::stoi(argv[1])));
	int ITER = std::stoi(argv[3]);
	cublasHandle_t handle;
	cublasCreate(&handle);
	double* newa = new double[GRID_SIZE * GRID_SIZE];
	double* olda = new double[GRID_SIZE * GRID_SIZE];
	double* dif = new double[GRID_SIZE * GRID_SIZE];
	int iter_count = 0;
	double error = 1.0;
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
	clock_t beforeinit = clock();

#pragma acc enter data copyin(error, olda[0:(GRID_SIZE * GRID_SIZE)], newa[0:(GRID_SIZE * GRID_SIZE)], dif[0:(GRID_SIZE * GRID_SIZE)])
	{

#pragma acc data present(newa, olda)
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
#pragma acc wait(1) async(2)
		std::cout << "Initialization time: " << 1.0 * (clock() - beforeinit) / CLOCKS_PER_SEC << std::endl;
		clock_t beforecal = clock();
		while (iter_count < ITER && error > ACC) {
			iter_count++;
			if (iter_count % 100 == 0) {
				error = 0.000001;
#pragma acc update device(error) async(2)
			}

#pragma acc data present(newa, olda, error)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max:error) async(2)
			for (size_t i = 1; i < GRID_SIZE - 1; i++) {
				for (size_t j = 1; j < GRID_SIZE - 1; j++) {
					newa[i * GRID_SIZE + j] = 0.25 * (olda[(i + 1) * GRID_SIZE + j] + olda[(i - 1) * GRID_SIZE + j] + olda[i * GRID_SIZE + j - 1] + olda[i * GRID_SIZE + j + 1]);
				}
			}
			int index = 0;
#pragma acc host_data use_device(newa, olda, dif)
			{
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, GRID_SIZE, GRID_SIZE, GRID_SIZE, 1.0, newa,  GRID_SIZE, olda, GRID_SIZE, -1.0, dif, GRID_SIZE)
			cublasIdamax(handle, GRID_SIZE, dif, 1, &index);
			error = dif[index-1];
			}
			if (iter_count % 100 == 0) {

#pragma acc update host(error) async(2)

#pragma acc wait(2)
			}
			double* c = olda;
			olda = newa;
			newa = c;
		}
		std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;
	}
	std::cout << "Iteration: " << iter_count << " " << "Error: " << error << std::endl;
	return 0;
}