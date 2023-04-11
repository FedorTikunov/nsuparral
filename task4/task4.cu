
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CORN1 10.0
#define CORN2 20.0
#define CORN3 30.0
#define CORN4 20.0



__global__ void calculationMatrix(double* new_arry, const double* old_array)
{
    /*
    int blockIndex = blockIdx.x + gridDim.y * blockIdx.y;
    int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;


    int arrayIndex = blockIndex * blockDim.x * blockDim.y + threadIndex;
    int  GRID_SIZEX = gridDim.x * blockDim.x;
    int  GRID_SIZEY = gridDim.y * blockDim.y;
    int i = arrayIndex / GRID_SIZEX;
    int j = arrayIndex % GRID_SIZEX;
    if (i != 0 && i != GRID_SIZEY - 1 && j != 0 && j != GRID_SIZEX - 1) {
        new_arry[i * GRID_SIZEX + j] = 0.25 * (old_array[(i + 1) * GRID_SIZEX + j] + old_array[(i - 1) * GRID_SIZEX + j] + old_array[i * GRID_SIZEX + j - 1] + old_array[i * GRID_SIZEX + j + 1]);
        //printf("(%lf - %lf)\n", old_array[i * GRID_SIZEX + j], new_arry[i * GRID_SIZEX + j]);
    }
    */
    size_t i = blockIdx.x;
    size_t j = threadIdx.x;

    if (if (i != 0 && i != GRID_SIZEY - 1 && j != 0 && j != GRID_SIZEX - 1))
    {
        new_arry[i * size + j] = 0.25 * (old_array[i * size + j - 1] + old_array[(i - 1) * size + j] +
            old_array[(i + 1) * size + j] + old_array[i * size + j + 1]);
    }
}

__global__ void getDifferenceMatrix(const double* new_arry, const double* old_array, double* dif)
{
    int blockIndex = blockIdx.x + gridDim.y * blockIdx.y;
    int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;


    int arrayIndex = blockIndex * blockDim.x * blockDim.y + threadIndex;
    int  GRID_SIZEX = gridDim.x * blockDim.x;
    int  GRID_SIZEY = gridDim.y * blockDim.y;
    int i = arrayIndex / GRID_SIZEX;
    int j = arrayIndex % GRID_SIZEX;
    if (i != 0 && i != GRID_SIZEY - 1 && j != 0 && j != GRID_SIZEX - 1) {
        //printf("%lf = abs(%lf - %lf)\n", dif[i * GRID_SIZEX + j], old_array[i * GRID_SIZEX + j], new_arry[i * GRID_SIZEX + j]);
        dif[i * GRID_SIZEX + j] = std::abs(old_array[i * GRID_SIZEX + j] - new_arry[i * GRID_SIZEX + j]);
    }
}


int main(int argc, char** argv) {
    int GRID_SIZE = std::stoi(argv[2]);
    double ACC = std::pow(10, -(std::stoi(argv[1])));
    int ITER = std::stoi(argv[3]);
    double* newa = new double[GRID_SIZE * GRID_SIZE];
    double* olda = new double[GRID_SIZE * GRID_SIZE];


    std::memset(olda, 0, GRID_SIZE * GRID_SIZE * sizeof(double));

    int iter_count = 0;
    double error = 1.0;

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

    double* d_newa,* d_olda, *d_dif;
    cudaMalloc((void**)&d_olda, sizeof(double) * GRID_SIZE * GRID_SIZE);
    cudaMalloc((void**)&d_newa, sizeof(double) * GRID_SIZE * GRID_SIZE);
    cudaMalloc((void**)&d_dif, sizeof(double) * GRID_SIZE * GRID_SIZE);


    clock_t beforeinit = clock();
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

    dim3 block_dim(32, 32);
    dim3 grid_dim(GRID_SIZE / block_dim.x, GRID_SIZE/ block_dim.y);

    cudaMemcpy(d_olda, olda, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_newa, newa, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice);
    double* max_error = 0;
    cudaMalloc((void**)&max_error, sizeof(double));
    std::cout << "Initialization time: " << 1.0 * (clock() - beforeinit) / CLOCKS_PER_SEC << std::endl;
    size_t temp_storage_bytes = 0;
    double* temp_storage = NULL;
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, GRID_SIZE * GRID_SIZE);
    cudaMalloc((void**)&temp_storage, temp_storage_bytes);

    clock_t beforecal = clock();
    double beta = -1.0;
    while (iter_count < ITER && error > ACC) {
        iter_count++;
        calculationMatrix <<<GRID_SIZE-1, GRID_SIZE-1>>> (d_newa, d_olda);

        if(iter_count % 100 == 0){
            getDifferenceMatrix <<<grid_dim, block_dim >>> (d_newa, d_olda, d_dif);
            cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, GRID_SIZE * GRID_SIZE);
            cudaMemcpy(&error, max_error, sizeof(double), cudaMemcpyDeviceToHost);
            error = std::abs(error);
        }

        double* c = d_olda;
        d_olda = d_newa;
        d_newa = c;
    }
    
    /*
    cudaMemcpy((void*)olda, (void*)d_olda, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < GRID_SIZE; i++)
    {
        for (size_t j = 0; j < GRID_SIZE; j++)
        {
            std::cout << olda[i * GRID_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    */

    std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Iteration: " << iter_count << " " << "Error: " << error << std::endl;

    cudaFree(d_olda);
    cudaFree(d_newa);
    cudaFree(temp_storage);
    delete[] olda;
    delete[] newa;
return 0;
}

