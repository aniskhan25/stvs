
/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation. 
* Any use, reproduction, disclosure, or distribution of this software 
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement( EULA) 
* associated with this source code for terms and conditions that govern 
* your use of this NVIDIA software.
* 
*/
/*
Parallel reduction kernels
*/
#ifndef _REDUCTION_CU_
#define _REDUCTION_CU_

#include <stdio.h>
#include "STA_Reduce.hpp"

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

bool isPow2( unsigned int x)
{
	return(( x&( x-1))==0);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator       T*()
	{
		extern __shared__ int __smem[];
		return( T*)__smem;
	}

	__device__ inline operator const T*() const
	{
		extern __shared__ int __smem[];
		return( T*)__smem;
	}
};

__global__ void Static::reduce6_sum_kernel( float *g_idata, float *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2)
{
	// now that we are using warp-synchronous programming( below)
	// we need to declare our shared memory volatile so that the compiler
	// doesn't reorder stores to it and induce incorrect behavior.
	volatile float *sdata = SharedMemory<float>();

	// perform first level of reduction, 
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks( via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while( i < n)
	{         
		mySum += g_idata[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if( nIsPow2 || i + blockSize < n) 
			mySum += g_idata[i+blockSize];  
		i += gridSize;
	} 

	// each thread puts its local sum into shared memory 
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if( blockSize >= 512) { if( tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
	if( blockSize >= 256) { if( tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
	if( blockSize >= 128) { if( tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
	if( tid < 32)
#endif
	{
		if( blockSize >=  64) { sdata[tid] = mySum = mySum + sdata[tid + 32]; EMUSYNC; }
		if( blockSize >=  32) { sdata[tid] = mySum = mySum + sdata[tid + 16]; EMUSYNC; }
		if( blockSize >=  16) { sdata[tid] = mySum = mySum + sdata[tid +  8]; EMUSYNC; }
		if( blockSize >=   8) { sdata[tid] = mySum = mySum + sdata[tid +  4]; EMUSYNC; }
		if( blockSize >=   4) { sdata[tid] = mySum = mySum + sdata[tid +  2]; EMUSYNC; }
		if( blockSize >=   2) { sdata[tid] = mySum = mySum + sdata[tid +  1]; EMUSYNC; }
	}

	// write result for this block to global mem 
	if( tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

void Static::Reduce::reduce6_sum( int size, int threads, int blocks, float *d_idata, float *d_odata)
{
	dim3 dimBlock( threads, 1, 1);
	dim3 dimGrid( blocks, 1, 1);
	int smemSize =( threads <= 32) ? 2 * threads * sizeof( float) : threads * sizeof( float);

	if( isPow2( size))
	{
		switch( threads)
		{
		case 512:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 512, true); break;
		case 256:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 256, true); break;
		case 128:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 128, true); break;
		case 64:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 64, true); break;
		case 32:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 32, true); break;
		case 16:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 16, true); break;
		case  8:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 8, true); break;
		case  4:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 4, true); break;
		case  2:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 2, true); break;
		case  1:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 1, true); break;
		}
	}
	else
	{
		switch( threads)
		{
		case 512:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 512, false); break;
		case 256:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 256, false); break;
		case 128:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 128, false); break;
		case 64:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 64, false); break;
		case 32:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 32, false); break;
		case 16:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 16, false); break;
		case  8:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 8, false); break;
		case  4:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 4, false); break;
		case  2:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 2, false); break;
		case  1:
			reduce6_sum_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 1, false); break;
		}
	}
}


/////////////////////////////////////////

__global__ void Static::reduce6_max_kernel( float *g_idata, float *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2)
{
	// now that we are using warp-synchronous programming( below)
	// we need to declare our shared memory volatile so that the compiler
	// doesn't reorder stores to it and induce incorrect behavior.
	volatile float *sdata = SharedMemory<float>();

	// perform first level of reduction, 
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float maxi = g_idata[i];

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks( via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while( i < n)
	{         
		if( g_idata[i] > maxi)
			maxi = g_idata[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if( nIsPow2 || i + blockSize < n) 
			if( g_idata[i+blockSize] > maxi)
				maxi = g_idata[i+blockSize];  
		i += gridSize;
	} 

	// each thread puts its local sum into shared memory 
	sdata[tid] = maxi;
	__syncthreads();


	// do reduction in shared mem
	if( blockSize >= 512) { if( tid < 256) { if( maxi < sdata[tid + 256]) sdata[tid] = maxi = sdata[tid + 256]; } __syncthreads(); }
	if( blockSize >= 256) { if( tid < 128) { if( maxi < sdata[tid + 128]) sdata[tid] = maxi = sdata[tid + 128]; } __syncthreads(); }
	if( blockSize >= 128) { if( tid <  64) { if( maxi < sdata[tid + 64]) sdata[tid] = maxi = sdata[tid + 64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
	if( tid < 32)
#endif
	{
		if( blockSize >=  64) { if( maxi < sdata[tid + 32]) sdata[tid] = maxi = sdata[tid + 32]; EMUSYNC; }		
		if( blockSize >=  32) { if( maxi < sdata[tid + 16]) sdata[tid] = maxi = sdata[tid + 16]; EMUSYNC; }
		if( blockSize >=  16) { if( maxi < sdata[tid + 8]) sdata[tid] = maxi = sdata[tid + 8]; EMUSYNC; }
		if( blockSize >=  8) { if( maxi < sdata[tid + 4]) sdata[tid] = maxi = sdata[tid + 4]; EMUSYNC; }
		if( blockSize >=  4) { if( maxi < sdata[tid + 2]) sdata[tid] = maxi = sdata[tid + 2]; EMUSYNC; }
		if( blockSize >=  2) { if( maxi < sdata[tid + 1]) sdata[tid] = maxi = sdata[tid + 1]; EMUSYNC; }		
	}

	// write result for this block to global mem 
	if( tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

void Static::Reduce::reduce6_max( int size, int threads, int blocks, float *d_idata, float *d_odata)
{
	dim3 dimBlock( threads, 1, 1);
	dim3 dimGrid( blocks, 1, 1);
	int smemSize =( threads <= 32) ? 2 * threads * sizeof( float) : threads * sizeof( float);

	if( isPow2( size))
	{
		switch( threads)
		{
		case 512:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 512, true); break;
		case 256:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 256, true); break;
		case 128:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 128, true); break;
		case 64:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 64, true); break;
		case 32:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 32, true); break;
		case 16:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 16, true); break;
		case  8:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 8, true); break;
		case  4:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 4, true); break;
		case  2:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 2, true); break;
		case  1:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 1, true); break;
		}
	}
	else
	{
		switch( threads)
		{
		case 512:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 512, false); break;
		case 256:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 256, false); break;
		case 128:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 128, false); break;
		case 64:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 64, false); break;
		case 32:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 32, false); break;
		case 16:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 16, false); break;
		case  8:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 8, false); break;
		case  4:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 4, false); break;
		case  2:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 2, false); break;
		case  1:
			reduce6_max_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 1, false); break;
		}
	}
}


/////////////////////////////////////////////////////////

__global__ void Static::reduce6_min_kernel( float *g_idata, float *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2)
{
	// now that we are using warp-synchronous programming( below)
	// we need to declare our shared memory volatile so that the compiler
	// doesn't reorder stores to it and induce incorrect behavior.
	volatile float *sdata = SharedMemory<float>();

	// perform first level of reduction, 
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mini = g_idata[i];

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks( via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while( i < n)
	{         
		if( g_idata[i] < mini)
			mini = g_idata[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if( nIsPow2 || i + blockSize < n) 
			if( g_idata[i+blockSize] < mini)
				mini = g_idata[i+blockSize];  
		i += gridSize;
	} 

	// each thread puts its local sum into shared memory 
	sdata[tid] = mini;
	__syncthreads();


	// do reduction in shared mem
	if( blockSize >= 512) { if( tid < 256) { if( mini > sdata[tid + 256]) sdata[tid] = mini = sdata[tid + 256]; } __syncthreads(); }
	if( blockSize >= 256) { if( tid < 128) { if( mini > sdata[tid + 128]) sdata[tid] = mini = sdata[tid + 128]; } __syncthreads(); }
	if( blockSize >= 128) { if( tid <  64) { if( mini > sdata[tid + 64]) sdata[tid] = mini = sdata[tid + 64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
	if( tid < 32)
#endif
	{
		if( blockSize >=  64) { if( mini > sdata[tid + 32]) sdata[tid] = mini = sdata[tid + 32]; EMUSYNC; }		
		if( blockSize >=  32) { if( mini > sdata[tid + 16]) sdata[tid] = mini = sdata[tid + 16]; EMUSYNC; }
		if( blockSize >=  16) { if( mini > sdata[tid + 8]) sdata[tid] = mini = sdata[tid + 8]; EMUSYNC; }
		if( blockSize >=  8) { if( mini > sdata[tid + 4]) sdata[tid] = mini = sdata[tid + 4]; EMUSYNC; }
		if( blockSize >=  4) { if( mini > sdata[tid + 2]) sdata[tid] = mini = sdata[tid + 2]; EMUSYNC; }
		if( blockSize >=  2) { if( mini > sdata[tid + 1]) sdata[tid] = mini = sdata[tid + 1]; EMUSYNC; }		
	}

	// write result for this block to global mem 
	if( tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

void Static::Reduce::reduce6_min( int size, int threads, int blocks, float *d_idata, float *d_odata)
{
	dim3 dimBlock( threads, 1, 1);
	dim3 dimGrid( blocks, 1, 1);
	int smemSize =( threads <= 32) ? 2 * threads * sizeof( float) : threads * sizeof( float);

	if( isPow2( size))
	{
		switch( threads)
		{
		case 512:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 512, true); break;
		case 256:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 256, true); break;
		case 128:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 128, true); break;
		case 64:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 64, true); break;
		case 32:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 32, true); break;
		case 16:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 16, true); break;
		case  8:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 8, true); break;
		case  4:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 4, true); break;
		case  2:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 2, true); break;
		case  1:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 1, true); break;
		}
	}
	else
	{
		switch( threads)
		{
		case 512:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 512, false); break;
		case 256:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 256, false); break;
		case 128:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 128, false); break;
		case 64:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 64, false); break;
		case 32:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 32, false); break;
		case 16:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 16, false); break;
		case  8:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 8, false); break;
		case  4:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 4, false); break;
		case  2:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 2, false); break;
		case  1:
			reduce6_min_kernel<<< dimGrid, dimBlock, smemSize >>>( d_idata, d_odata, size, 1, false); break;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void Static::Reduce::getNumBlocksAndThreads( int n, int maxThreads, int &blocks, int &threads)
{
	if( n == 1) 
		threads = 1;
	else
		threads =( n < maxThreads*2) ? n / 2 : maxThreads;

	blocks = n /( threads * 2);
}

#endif // #ifndef _REDUCTION_CU_