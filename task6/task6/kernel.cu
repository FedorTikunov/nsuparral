
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <fstream>
#include <cmath>
#include <iostream>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}





#define checkCUDNN(expression)                             \
{                                                           \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
        std::cerr << "Error on line " << __LINE__ << ": "   \
                  << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                            \
    }                                                       \
}

#define checkCublas(expression)                             \
{                                                           \
    cublasStatus_t status = (expression);                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                  \
        std::cerr << "Error on line " << __LINE__ << ": "   \
                  << _cudaGetErrorEnum(status) << std::endl; \
        std::exit(EXIT_FAILURE);                            \
    }                                                       \
}

int main(int argc, char const* argv[]) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    const int batchSize = 1;
    const int inputSize = 32 * 32;
    const int hiddenSize1 = 16 * 16;
    const int hiddenSize2 = 4 * 4;
    const int outputSize = 1;

    // Input layer
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        inputSize,
        1,
        1));

    // Hidden layer 1
    cudnnTensorDescriptor_t hidden_descriptor1;
    checkCUDNN(cudnnCreateTensorDescriptor(&hidden_descriptor1));
    checkCUDNN(cudnnSetTensor4dDescriptor(hidden_descriptor1,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        hiddenSize1,
        1,
        1));

    // Hidden layer 2
    cudnnTensorDescriptor_t hidden_descriptor2;
    checkCUDNN(cudnnCreateTensorDescriptor(&hidden_descriptor2));
    checkCUDNN(cudnnSetTensor4dDescriptor(hidden_descriptor2,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        hiddenSize2,
        1,
        1));

    // Output layer
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        outputSize,
        1,
        1));

    // Activation descriptor
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
        CUDNN_ACTIVATION_SIGMOID,
        CUDNN_PROPAGATE_NAN,
        0.0));

    // Allocate device memory for input, hidden, and output layers
    float* d_input{ nullptr }, * d_hidden1{ nullptr }, * d_hidden2{ nullptr }, * d_output{ nullptr };
    cudaMalloc(&d_input, sizeof(float) * inputSize);
    cudaMalloc(&d_hidden1, sizeof(float) * hiddenSize1);
    cudaMalloc(&d_hidden2, sizeof(float) * hiddenSize2);
    cudaMalloc(&d_output, sizeof(float) * outputSize);

    // Allocate device memory for weights and biases
    float* d_w1{ nullptr }, * d_w2{ nullptr }, * d_w3{ nullptr };
    float* d_b1{ nullptr }, * d_b2{ nullptr }, * d_b3{ nullptr };
    cudaMalloc(&d_w1, sizeof(float) * inputSize * hiddenSize1);
    cudaMalloc(&d_w2, sizeof(float) * hiddenSize1 * hiddenSize2);
    cudaMalloc(&d_w3, sizeof(float) * hiddenSize2 * outputSize);
    cudaMalloc(&d_b1, sizeof(float) * hiddenSize1);
    cudaMalloc(&d_b2, sizeof(float) * hiddenSize2);
    cudaMalloc(&d_b3, sizeof(float) * outputSize);
    // Allocate host memory for weights and biases
    float* h_w1{ nullptr }, * h_w2{ nullptr }, * h_w3{ nullptr };
    float* h_b1{ nullptr }, * h_b2{ nullptr }, * h_b3{ nullptr };
    float *h_input{ nullptr };
    float* h_output{ nullptr };

    h_input = (float*)malloc(sizeof(float) * inputSize);
    h_output = (float*)malloc(sizeof(float) * outputSize);

    h_w1 = (float*)malloc(sizeof(float) * inputSize * hiddenSize1);
    h_w2 = (float*)malloc(sizeof(float) * hiddenSize1 * hiddenSize2);
    h_w3 = (float*)malloc(sizeof(float) * hiddenSize2 * outputSize);
    h_b1 = (float*)malloc(sizeof(float) * hiddenSize1);
    h_b2 = (float*)malloc(sizeof(float) * hiddenSize2);
    h_b3 = (float*)malloc(sizeof(float) * outputSize);

    // Initialize weights and biases
    std::ifstream wb_file("wb.bin", std::ios::binary);
    if (!wb_file) {
        std::cerr << "Failed to open wb file" << std::endl;
        return -1;
    }
    wb_file.read(reinterpret_cast<char*>(h_w1), sizeof(float) * inputSize * hiddenSize1);
    wb_file.read(reinterpret_cast<char*>(h_w2), sizeof(float) * hiddenSize1 * hiddenSize2);
    wb_file.read(reinterpret_cast<char*>(h_w3), sizeof(float) * hiddenSize2 * outputSize);

    wb_file.read(reinterpret_cast<char*>(h_b1), sizeof(float) * hiddenSize1);
    wb_file.read(reinterpret_cast<char*>(h_b2), sizeof(float) * hiddenSize2);
    wb_file.read(reinterpret_cast<char*>(h_b3), sizeof(float) * outputSize);

    wb_file.close();

    cudaMemcpy(d_w1, h_w1, sizeof(float) * inputSize * hiddenSize1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, sizeof(float) * hiddenSize1 * hiddenSize2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3, sizeof(float) * hiddenSize2 * outputSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_b1, h_b1, sizeof(float) * hiddenSize1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, sizeof(float) * hiddenSize2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, sizeof(float) * outputSize, cudaMemcpyHostToDevice);

    // Initialize input data
    std::ifstream input_file("input.bin", std::ios::binary);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        return -1;
    }
    input_file.read(reinterpret_cast<char*>(h_input), sizeof(float) * inputSize);

    input_file.close();

    cudaMemcpy(d_input, h_input, sizeof(float) * inputSize, cudaMemcpyHostToDevice);

    // Forward pass
    const float alpha = 1.0f;
    const float beta = 0.0f;
    clock_t beforecal = clock();
    // Input -> Hidden 1
    cublasSgemv(cublas, CUBLAS_OP_T, inputSize, hiddenSize1, &alpha, d_w1, inputSize, d_input, 1, &beta, d_hidden1,1);
    cublasSaxpy(cublas, hiddenSize1, &alpha, d_b1, 1, d_hidden1, 1);
    cudnnActivationForward(cudnn, activation_descriptor, &alpha, hidden_descriptor1, d_hidden1, &beta, hidden_descriptor1, d_hidden1);

    // Hidden 1 -> Hidden 2
    cublasSgemv(cublas, CUBLAS_OP_T, hiddenSize1, hiddenSize2, &alpha, d_w2, hiddenSize1, d_hidden1, 1, &beta, d_hidden2, 1);
    cublasSaxpy(cublas, hiddenSize2, &alpha, d_b2, 1, d_hidden2, 1);
    cudnnActivationForward(cudnn, activation_descriptor, &alpha, hidden_descriptor2, d_hidden2, &beta, hidden_descriptor2, d_hidden2);

    // Hidden 2 -> Output
    cublasSgemv(cublas, CUBLAS_OP_T, hiddenSize2, outputSize, &alpha, d_w3, hiddenSize2, d_hidden2, 1, &beta, d_output, 1);
    cublasSaxpy(cublas, outputSize, &alpha, d_b3, 1, d_output, 1);
    cudnnActivationForward(cudnn, activation_descriptor, &alpha, output_descriptor, d_output, &beta, output_descriptor, d_output);

    std::cout << "Calculation time: " << 1.0 * (clock() - beforecal) / CLOCKS_PER_SEC << std::endl;

    // Copy result from device to host an print
    cudaMemcpy(h_output, d_output, sizeof(float) * outputSize, cudaMemcpyDeviceToHost);
    std::cout << h_output[0] << std::endl;
    std::ofstream output_file("output.bin", std::ios::binary);
    if (!output_file) {
        std::cerr << "Failed to open output file" << std::endl;
        return -1;
    }

    output_file.write(reinterpret_cast<char*>(h_output), sizeof(float) * outputSize);

    output_file.close();

    // Cleanup

    cudaFree(h_input);
    cudaFree(h_output);
    cudaFree(h_w1);
    cudaFree(h_w2);
    cudaFree(h_w3);
    cudaFree(h_b1);
    cudaFree(h_b2);
    cudaFree(h_b3);

    cudaFree(d_input);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_w3);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);

    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(hidden_descriptor1);
    cudnnDestroyTensorDescriptor(hidden_descriptor2);
    cudnnDestroyTensorDescriptor(output_descriptor);

    cudnnDestroy(cudnn);

    return 0;
}