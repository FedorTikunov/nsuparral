
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <mpi.h>

#define WIDTH 80
#define HEIGHT 24

#define BLOCK_SIZE 32
#define GENERATIONS 100

__global__ void update(int* grid, int* newGrid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int index = y * width + x;
    int neighbors = 0;
    int x1, x2, y1, y2;

    x1 = (x == 0) ? width - 1 : x - 1;
    x2 = (x == width - 1) ? 0 : x + 1;
    y1 = (y == 0) ? height - 1 : y - 1;
    y2 = (y == height - 1) ? 0 : y + 1;

    neighbors += grid[y1 * width + x1];
    neighbors += grid[y1 * width + x];
    neighbors += grid[y1 * width + x2];
    neighbors += grid[y * width + x1];
    neighbors += grid[y * width + x2];
    neighbors += grid[y2 * width + x1];
    neighbors += grid[y2 * width + x];
    neighbors += grid[y2 * width + x2];

    if (grid[index] == 1 && (neighbors < 2 || neighbors > 3)) {
        newGrid[index] = 0;
    }
    else if (grid[index] == 0 && neighbors == 3) {
        newGrid[index] = 1;
    }
    else {
        newGrid[index] = grid[index];
    }
}

void printGrid(int* grid, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << (grid[y * width + x] ? '#' : ' ');
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = WIDTH;
    int height = HEIGHT / size;
    int* grid = new int[width * height];
    int* newGrid = new int[width * height];

    srand(time(NULL));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            grid[y * width + x] = rand() % 2;
        }
    }

    int* d_grid, * d_newGrid;
    cudaMalloc(&d_grid, width * height * sizeof(int));
    cudaMalloc(&d_newGrid, width * height * sizeof(int));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MPI_Request reqs[4];
    MPI_Status stats[4];
    int up, down;

    if (rank == 0) {
        up = MPI_PROC_NULL;
        down = 1;
    }
    else if (rank == size - 1) {
        up = rank - 1;
        down = MPI_PROC_NULL;
    }
    else {
        up = rank - 1;
        down = rank + 1;
    }

    for (int i = 0; i < GENERATIONS; i++) {
        cudaMemcpy(d_grid, grid, width * height * sizeof(int), cudaMemcpyHostToDevice);

        update << <numBlocks, threadsPerBlock >> > (d_grid, d_newGrid, width, height);

        cudaMemcpy(newGrid, d_newGrid, width * height * sizeof(int), cudaMemcpyDeviceToHost);

        MPI_Isend(newGrid, width * sizeof(int) * height, MPI_CHAR, up, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(newGrid + (height - 1) * width, width * sizeof(int), MPI_CHAR, down, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(grid + (height + 1) * width, width * sizeof(int), MPI_CHAR, down, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(grid, width * sizeof(int), MPI_CHAR, up, 1, MPI_COMM_WORLD, &reqs[3]);

        MPI_Waitall(4, reqs, stats);

        int* temp = grid;
        grid = newGrid;
        newGrid = temp;

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            system("clear");
            printGrid(grid, width, height * size);
        }
    }

    MPI_Finalize();

    delete[] grid;
    delete[] newGrid;

    cudaFree(d_grid);
    cudaFree(d_newGrid);

    return 0;
}