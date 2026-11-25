/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis, Sebastian Hahn

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//#include "LatticeFilterKernel.h"

#include <cassert>

#include <iostream>

#include "PermutohedralLatticeGPU.cuh"
#include "DeviceMemoryAllocator.h"
#include "../Devices.hpp"
#ifndef SPATIAL_DIMS
#define SPATIAL_DIMS 2
#endif
#ifndef INPUT_CHANNELS
#define INPUT_CHANNELS 3
#endif
#ifndef REFERENCE_CHANNELS
#define REFERENCE_CHANNELS 3
#endif
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void computeKernelGPU(
    const float* input_image,
    float* positions,
    int num_super_pixels,
    int n_spatial_dims,
    int *spatial_dims,
    int n_reference_channels,
    float spatial_std,
    float features_std
)
{

    // printf("spatial_std = %f, n_spatial_dims = %d, spatial_dims[0] = %d, spatial_dims[1] = %d\n",
    //         spatial_std, n_spatial_dims, spatial_dims[0], spatial_dims[1]);

    int* spatial_dims_gpu;
    
    gpuErrchk(
        cudaMalloc(
            (void**)&spatial_dims_gpu,
            n_spatial_dims*sizeof(int)
        )
    );

    gpuErrchk(
        cudaMemcpy(
            spatial_dims_gpu,
            spatial_dims,
            n_spatial_dims*sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    // auto allocator = DeviceMemoryAllocator(context);
    // allocator.allocate_device_memory<int>((void**)&spatial_dims_gpu, n_spatial_dims);
    // gpuErrchk(cudaMemcpy(spatial_dims_gpu, spatial_dims, n_spatial_dims*sizeof(int), cudaMemcpyHostToDevice));

    // // DEBUG: Retrieves data from device and print
    // int* spatial_dims_copy = (int*) malloc(n_spatial_dims*sizeof(int));
    // // cudaMalloc((void**)&spatial_dims_copy, n_spatial_dims*sizeof(int));
    // cudaMemcpy(spatial_dims_copy, spatial_dims_gpu, n_spatial_dims*sizeof(int), cudaMemcpyDeviceToHost);
    // printf("spatial_dims_copy[0] = %d, spatial_dims_copy[1] = %d\n", spatial_dims_copy[0], spatial_dims_copy[1]);

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //return input_tensor;

    // std::cout<<"positions before: \n"<<positions<<std::endl;
    // std::cout<<"image: \n"<<input_image<<std::endl;

    compute_kernel<float><<<blocks, blockSize>>>(
        input_image,
        positions,
        num_super_pixels,
        n_reference_channels,
        n_spatial_dims,
        spatial_dims_gpu,
        spatial_std,
        features_std
    );

    // std::cout<<"positions after: \n"<<positions<<std::endl;

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(
        cudaFree(
            spatial_dims_gpu
        )
    );
};

//declaration of what lattices (pd and vd) can be used


// Define the GPU implementation that launches the CUDA kernel.
template<int pd,int vd> void latticeFilterGPU(
    float* output,
    const float* input,
    const float* positions,
    int num_super_pixels,
    bool reverse)
{
    auto allocator = DeviceMemoryAllocator();
    //bilateral

    auto lattice = PermutohedralLatticeGPU<float,pd,vd>(num_super_pixels, &allocator);

    lattice.filter(output, input, positions,reverse);
}


extern "C" void insid3_filter(float* out,const float* input,const float* image,float* pos,int h,int w,int bilateral,float alpha,float beta,float gamma) {
    int dims[2]={h,w};
    computeKernelGPU(image,pos,h*w,2,dims,bilateral?3:0,bilateral?alpha:gamma,bilateral?beta:-1);
    if(bilateral) latticeFilterGPU<5,3>(out,input,pos,h*w,false);
    else latticeFilterGPU<2,3>(out,input,pos,h*w,false);
}
