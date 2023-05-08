
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iomanip>

#include "mpi.h"
#include "nccl.h"

//значения в углах сетки
#define CORN1 10.0
#define CORN2 20.0
#define CORN3 30.0
#define CORN4 20.0


//функция по подсчету/обновлению ячейк сетке
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//функция получает указатели двух массив. 
//обновляет ячейки первого массива на основе среднего значения четерех ближайших по индексу ячейк из второго массива
//функция являеться global и распоточивает подсчет матрицы на потоки
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calculationMatrix(double* new_arry, const double* old_array, size_t size, size_t groupSize)
{
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("%d", size);
    if (i > 0 && i < groupSize - 1 && j > 0 && j < size - 1)
    {
        new_arry[i * size + j] = 0.25 * (old_array[i * size + j - 1] + old_array[(i - 1) * size + j] +
            old_array[(i + 1) * size + j] + old_array[i * size + j + 1]);
    }
}
//функция по вычислению разницы матриц
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//функция получает указатели трех массив. 
//модуль разницы двух первых массивов записывает в третий
//при распоточивание, 1d массивы разбеваются на блоки по 32x32 как 2d массивы
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getDifferenceMatrix(const double* new_arry, const double* old_array, double* dif)
{   
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    dif[idx] = std::abs(old_array[idx] - new_arry[idx]);
}

//основной код
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//получает из коммандной строки значения для размерность сетки, точности обновления сетки, максимального количества итераций
//выделяет память на host и device для сеток
//заполняем сетки начальными значениями
//производим вычисления на GPU
//выводим скорость вычисления, кол. итераций и точнось в консоль
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    // Получаем значения из коммандной строки
    int GRID_SIZE = std::stoi(argv[2]); // размерность сетки
    double ACC = std::pow(10, -(std::stoi(argv[1]))); // до какой точность обновлять сетку
    int ITER = std::stoi(argv[3]); //  максимальное количество итераций


    int rank, sizeOfTheGroup;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeOfTheGroup);

    
    cudaSetDevice(rank);

    if (rank!=0)
        cudaDeviceEnablePeerAccess(rank - 1, 0);
    if (rank!=sizeOfTheGroup-1)
        cudaDeviceEnablePeerAccess(rank + 1, 0);

    ncclComm_t ncclcomm;
    ncclUniqueId idx;
    
    if (rank == 0) 
	{
		ncclGetUniqueId(&idx);
	}
    MPI_Bcast(&idx, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&ncclcomm, sizeOfTheGroup, idx, rank);

    size_t sizeOfAreaForOneProcess = GRID_SIZE / sizeOfTheGroup;
	size_t startYIdx = sizeOfAreaForOneProcess * rank;

    //выделяем память под 2 сетки размера GRID_SIZExGRID_SIZE
    double* newa, *olda;
    cudaMallocHost(&newa,  sizeof(double) * GRID_SIZE * GRID_SIZE); 
    cudaMallocHost(&olda,  sizeof(double) * GRID_SIZE * GRID_SIZE);

    std::memset(olda, 0, GRID_SIZE * GRID_SIZE * sizeof(double));


    int iter_count = 0; // счетчик итераций
    double* error;
	cudaMallocHost(&error, sizeof(double));
    *error = 1.0;

    double prop1 = (CORN2 - CORN1) / (GRID_SIZE);
    double prop2 = (CORN3 - CORN1) / (GRID_SIZE);
    double prop3 = (CORN4 - CORN3) / (GRID_SIZE);
    double prop4 = (CORN2 - CORN4) / (GRID_SIZE);

    //записываем значения в углы сеток
    olda[0] = CORN1;
    olda[(GRID_SIZE - 1) * GRID_SIZE] = CORN3;
    olda[GRID_SIZE - 1] = CORN2;
    olda[GRID_SIZE - 1 + GRID_SIZE * (GRID_SIZE - 1)] = CORN4;
    newa[0] = CORN1;
    newa[(GRID_SIZE - 1) * GRID_SIZE] = CORN3;
    newa[GRID_SIZE - 1] = CORN2;
    newa[GRID_SIZE - 1 + GRID_SIZE * (GRID_SIZE - 1)] = CORN4;


    //вычисления значений границ сетки
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

    if (rank != 0 && rank != sizeOfTheGroup - 1)
	{
		sizeOfAreaForOneProcess += 2;
	}
	else 
	{
		sizeOfAreaForOneProcess += 1;
	}
        
    size_t sizeOfAllocatedMemory = GRID_SIZE * sizeOfAreaForOneProcess;
        
    //выделяем память на gpu через cuda для 3 сеток
    double* d_newa,* d_olda, *d_dif;
    cudaMalloc((void**)&d_olda, sizeof(double) * sizeOfAllocatedMemory);
    cudaMalloc((void**)&d_newa, sizeof(double) * sizeOfAllocatedMemory);
    cudaMalloc((void**)&d_dif, sizeof(double) * sizeOfAllocatedMemory);

    unsigned int threads_x = (GRID_SIZE < 1024) ? GRID_SIZE : 1024;
    unsigned int blocks_y = sizeOfAreaForOneProcess;
    unsigned int blocks_x = GRID_SIZE / threads_x;

    dim3 blockDim1(threads_x, 1);
    dim3 gridDim1(blocks_x, blocks_y);

	size_t offset = (rank != 0) ? GRID_SIZE : 0;

    cudaMemset(d_olda, 0, sizeof(double) * sizeOfAllocatedMemory);
	cudaMemset(d_newa, 0, sizeof(double) * sizeOfAllocatedMemory);

    // CPU на GPU
    cudaMemcpy(d_olda, olda + (startYIdx * GRID_SIZE) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); // (CPU) olda -> (GPU) d_olda
    cudaMemcpy(d_newa, newa + (startYIdx * GRID_SIZE) - offset, sizeof(double) * sizeOfAllocatedMemory, cudaMemcpyHostToDevice); // (CPU) newa -> (GPU) d_newa


    //выделяем память на gpu для переменной, которая будет хранить ошибку на device
    double* max_error = 0;
    cudaMalloc((void**)&max_error, sizeof(double));
    if (rank == 0)
	{
        std::cout << "Initialization time: " << 1.0 * (clock() - beforeinit) / CLOCKS_PER_SEC << std::endl;
    }
    size_t temp_storage_bytes = 0;
    double* temp_storage = NULL;
    //получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, sizeOfAllocatedMemory);
    //выделяем память для буфера
    cudaMalloc((void**)&temp_storage, temp_storage_bytes);

    cudaStream_t stream, memoryStream;
    cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);

    clock_t beforecal = clock();
    
    //алгоритм обновления сетки, работающий пока макс. ошибка не станет меньше или равне нужной точности, или пока количество итерации не превысит максимальное количество.
    while (iter_count < ITER && (*error) > ACC) {
        iter_count += 1;
        //calculationMatrix <<<GRID_SIZE-1, GRID_SIZE-1>>> (d_newa, d_olda); // расчет матрицы
        calculationMatrix <<<gridDim1, blockDim1, 0, stream>>> (d_newa, d_olda, GRID_SIZE, sizeOfAreaForOneProcess); /// GRID_SIZE , sizeOfAreaForOneProcess);
        // Обмен верхней границей

        // расчитываем ошибку каждую сотую итерацию
        if(iter_count % 100 == 0){  
            getDifferenceMatrix <<<blocks_x * blocks_y, threads_x, 0, stream>>> (d_newa, d_olda, d_dif);
            cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, sizeOfAllocatedMemory); // нахождение максимума в разнице матрицы

            cudaStreamSynchronize(stream);
            
            ncclAllReduce((void*)max_error,(void*)max_error, 1, ncclDouble, ncclMax, ncclcomm, stream);
            
            cudaMemcpyAsync(error, max_error, sizeof(double), cudaMemcpyDeviceToHost, stream); // запись ошибки в переменную на host
            // Находим максимальную ошибку среди всех и передаём её всем процессам
			
        }     
        cudaStreamSynchronize(stream);

        ncclGroupStart();
        if (rank != 0)
		{
            ncclSend(d_newa + size + 1, size - 2, ncclDouble, rank - 1, comm, stream); 
            ncclRecv(d_newa + 1, size - 2, ncclDouble, rank - 1, comm, stream);

		}
        // Обмен нижней границей
		if (rank != sizeOfTheGroup - 1)
		{
            ncclSend(d_newa + (sizeOfAreaForOneProcess - 2) * size + 1, 
				size - 2, ncclDouble, rank + 1, comm, stream);
            ncclRecv(d_newa + (sizeOfAreaForOneProcess - 1) * size + 1, 
				size - 2, ncclDouble, rank + 1, comm, stream);
		}
        ncclGroupEnd();           
        double* c = d_olda; 
        d_olda = d_newa;
        d_newa = c;
    }
    
    if (rank == 0)
	{
        //вывод времени работы на алгоритма
        std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;
        //вывод кол. итерацций и значение ошибки
        std::cout << "Iteration: " << iter_count << " " << "Error: " << (*error) << std::endl;
    }
    //очитска памяти
    cudaFree(d_olda);
    cudaFree(d_newa);
    cudaFree(temp_storage);
    cudaFree(olda);
    cudaFree(newa);
    cudaFree(max_error);

    MPI_Finalize();
return 0;
}

