
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>


//значения в углах сетки
#define CORN1 10.0
#define CORN2 20.0
#define CORN3 30.0
#define CORN4 20.0

#define BlOCK_SIZE 16

//функция по подсчету/обновлению ячейк сетке
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//функция получает указатели двух массив. 
//обновляет ячейки первого массива на основе среднего значения четерех ближайших по индексу ячейк из второго массива
//функция являеться global и распоточивает подсчет матрицы на потоки
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calculationMatrix(double* new_arry, const double* old_array, size_t size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    //printf("%d", size);
    if (i != 0 && i != size - 1 && j != 0 && j != size - 1)
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

//основной код
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//получает из коммандной строки значения для размерность сетки, точности обновления сетки, максимального количества итераций
//выделяет память на host и device для сеток
//заполняем сетки начальными значениями
//производим вычисления на GPU
//выводим скорость вычисления, кол. итераций и точнось в консоль
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Получаем значения из коммандной строки
    int GRID_SIZE = std::stoi(argv[2]); // размерность сетки
    double ACC = std::pow(10, -(std::stoi(argv[1]))); // до какой точность обновлять сетку
    int ITER = std::stoi(argv[3]); //  максимальное количество итераций

    //выделяем память под 2 сетки размера GRID_SIZExGRID_SIZE
    double* newa = new double[GRID_SIZE * GRID_SIZE]; 
    double* olda = new double[GRID_SIZE * GRID_SIZE];


    std::memset(olda, 0, GRID_SIZE * GRID_SIZE * sizeof(double));

    int iter_count = 0; // счетчик итераций
    double error = 1.0; // переменная ошибки
    
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

    //выделяем память на gpu через cuda для 3 сеток
    double* d_newa,* d_olda, *d_dif;
    cudaMalloc((void**)&d_olda, sizeof(double) * GRID_SIZE * GRID_SIZE);
    cudaMalloc((void**)&d_newa, sizeof(double) * GRID_SIZE * GRID_SIZE);
    cudaMalloc((void**)&d_dif, sizeof(double) * GRID_SIZE * GRID_SIZE);

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
    
    // размерность блоков и грида 
    dim3 block_dim(32, 32);
    dim3 grid_dim(GRID_SIZE / block_dim.x, GRID_SIZE/ block_dim.y);

    // Define CUDA streams for overlapping computation and communication
    cudaStream_t compute_stream, comm_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&comm_stream);

    // Define CUDA grid and block sizes

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((GRID_SIZE - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_SIZE - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Define MPI data types for exchanging boundary conditions
    MPI_Datatype col_type;
    MPI_Type_vector(GRID_SIZE - 2, 1, N, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Datatype row_type;
    MPI_Type_contiguous(GRID_SIZE - 2, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);


    // копирование информации с CPU на GPU
    cudaMemcpy(d_olda, olda, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice); // (CPU) olda -> (GPU) d_olda
    cudaMemcpy(d_newa, newa, sizeof(double) * GRID_SIZE * GRID_SIZE, cudaMemcpyHostToDevice); // (CPU) newa -> (GPU) d_newa

    //выделяем память на gpu для переменной, которая будет хранить ошибку на device
    double* max_error = 0;
    cudaMalloc((void**)&max_error, sizeof(double));

    std::cout << "Initialization time: " << 1.0 * (clock() - beforeinit) / CLOCKS_PER_SEC << std::endl;

    size_t temp_storage_bytes = 0;
    double* temp_storage = NULL;
    //получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, GRID_SIZE * GRID_SIZE);

    //выделяем память для буфера
    cudaMalloc((void**)&temp_storage, temp_storage_bytes);

    clock_t beforecal = clock();
    
    //алгоритм обновления сетки, работающий пока макс. ошибка не станет меньше или равне нужной точности, или пока количество итерации не превысит максимальное количество.
    while (iter_count < ITER && error > ACC) {
        iter_count++;
        calculationMatrix << <grid, block, 0, compute_stream>>> (d_newa, d_olda, GRID_SIZE); // расчет матрицы

        // Wait for the computation to finish
        cudaStreamSynchronize(compute_stream);

        if (rank % 2 == 0) {
            MPI_Sendrecv(&d_newa[1 + GRID_SIZE * (GRID_SIZE - 2)], 1, col_type, rank + 1, 0,
                &d_newa[1 + GRID_SIZE * (GRID_SIZE - 1)], 1, col_type, rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&d_newa[1], 1, col_type, rank - 1, 0,
                &d_newa[0], 1, col_type, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Sendrecv(&d_newa[1], 1, col_type, rank - 1, 0,
                &d_newa[0], 1, col_type, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&d_newa[1 + GRID_SIZE], 1, col_type, rank + 1, 0,
                &d_newa[1 + GRID_SIZE * (GRID_SIZE - 1)], 1, col_type, rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&d_newa[GRID_SIZE * (GRID_SIZE - 2) + 1], GRID_SIZE - 2, MPI_FLOAT, rank + 1, 0,
                &d_newa[GRID_SIZE * (GRID_SIZE - 1) + 1], GRID_SIZE - 2, MPI_FLOAT, rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank > 0) {
            MPI_Sendrecv(&d_newa[GRID_SIZE + 1], GRID_SIZE - 2, MPI_FLOAT, rank - 1, 0,
                &d_newa[1], N - 2, MPI_FLOAT, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        // расчитываем ошибку каждую сотую итерацию
        if(iter_count % 100 == 0){
            getDifferenceMatrix <<<grid_dim, block_dim >>> (d_newa, d_olda, d_dif); // вычисления разницы матрицы
            cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_dif, max_error, GRID_SIZE * GRID_SIZE); // нахождение максимума в разнице матрицы
            cudaMemcpy(&error, max_error, sizeof(double), cudaMemcpyDeviceToHost); // запись ошибки в переменную на host

            error = std::abs(error);
        }

        //смена указателей между сетками на device
        double* c = d_olda;
        d_olda = d_newa;
        d_newa = c;
    }
    
    //вывод времени работы на алгоритма
    std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;
    //вывод кол. итерацций и значение ошибки
    std::cout << "Iteration: " << iter_count << " " << "Error: " << error << std::endl;

    //очитска памяти
    cudaFree(d_olda);
    cudaFree(d_newa);
    cudaFree(temp_storage);
    cudaFree(max_error);
    MPI_Type_free(&col_type);
    MPI_Type_free(&row_type);
    MPI_Finalize();

    delete[] olda;
    delete[] newa;
return 0;
}

