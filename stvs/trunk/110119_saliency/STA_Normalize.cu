#include "error.hpp"

#include "STA_Normalize.hpp"

/**
Normalization NL Kernel
*/
__global__  void Static::NormNLKernel( float* mapsout, float* maxpt, float* minpt, siz_t _im_size, int blocks)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float maximum, minimum, tmp ;

	if( x>=_im_size.w || y>=_im_size.h) return;

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			maximum = maxpt[( j*NO_OF_BANDS + i) * blocks];
			minimum = minpt[( j*NO_OF_BANDS + i) * blocks];

			tmp  = mapsout[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( y*_im_size.w + x)];

			// Normalize
			if( minimum != maximum)
			{
				tmp  -= minimum;
				tmp /=( maximum - minimum);
			}	

			mapsout[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( y*_im_size.w + x)] = tmp;
		}
	}
}

/**
Normalization Itti Kernel
*/
__global__  void Static::NormIttiKernel( float* mapsout, float* sumpt, float* maxpt, siz_t _im_size, int blocks)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float maximum, sum;

	if( x>=_im_size.w || y>=_im_size.h) return;

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			maximum = maxpt[( j*NO_OF_BANDS + i) * blocks];
			sum		= sumpt[( j*NO_OF_BANDS + i) * blocks]; // to reduce the effect of significant precision digits

			// Normalization Itti

			if( sum != maximum)
				mapsout[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( y*_im_size.w + x)] = mapsout[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( y*_im_size.w + x)]	*(( maximum - sum /( _im_size.w*_im_size.h)) *( maximum - sum /( _im_size.w*_im_size.h)));
		}
	}
}

/**
Normalization and Fusion Kernel
*/
__global__  void Static::NormPCFusionKernel( float* mapsout, float* maxpt, siz_t _im_size, float* imageout, int blocks)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=_im_size.w || y>=_im_size.h) return;

	extern __shared__ float buf[];	// based on no. of threads

	float maximum, tmp, level;

	buf[threadIdx.x] = 0.0f;

	__syncthreads();

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			maximum = maxpt[( j*NO_OF_BANDS + i) * blocks];

			tmp		= mapsout[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( y*_im_size.w + x)];

			// Normalization PC and Fusion
			level = 0.2f * maximum;

			if( tmp <= level) 
				tmp = 0.0f;
			else
				tmp =( tmp - level) /( maximum - level) * maximum;

			buf[threadIdx.x] += tmp;			

			__syncthreads();
		}
	}

	imageout[y*_im_size.w + x] = buf[threadIdx.x];	
}

void Static::Normalize::Apply( float* in, siz_t _im_size, float *out) {
	/**
	Reduction, normalizations, and fusion
	*/
	int s;
	int cpuFinalThreshold = 1;
	int numBlocks, numThreads;

	dim3 dimBlock( 64, 1, 1);	
	dim3 dimGrid(( _im_size.w / dimBlock.x) + 1*( _im_size.w%dimBlock.x!=0),( _im_size.h / dimBlock.y) + 1*( _im_size.h%dimBlock.y!=0), dimBlock.z);

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			numBlocks = 0;
			numThreads = 0;

			oReduce.getNumBlocksAndThreads( _im_size.w*_im_size.h, MAX_THREADS, numBlocks, numThreads);

			oReduce.reduce6_min( _im_size.w*_im_size.h, numThreads, numBlocks, &in[( j*NO_OF_BANDS + i)* _im_size.w*_im_size.h], &d_oMin[( j*NO_OF_BANDS + i)*numBlocks]);

			// sum partial block sums on GPU
			s=numBlocks;	

			while( s > cpuFinalThreshold) 
			{
				int threads = 0, blocks = 0;		
				oReduce.getNumBlocksAndThreads( s, MAX_THREADS, blocks, threads);

				oReduce.reduce6_min( s, threads, blocks, &d_oMin[( j*NO_OF_BANDS + i)*numBlocks], &d_oMin[( j*NO_OF_BANDS + i)*numBlocks]);

				s =( s +( threads*2-1)) /( threads*2);
			}
		}
	}

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			numBlocks = 0;
			numThreads = 0;

			oReduce.getNumBlocksAndThreads( _im_size.w*_im_size.h, MAX_THREADS, numBlocks, numThreads);

			oReduce.reduce6_max( _im_size.w*_im_size.h, numThreads, numBlocks, &in[( j*NO_OF_BANDS + i)* _im_size.w*_im_size.h], &d_oMax[( j*NO_OF_BANDS + i)*numBlocks]);

			// sum partial block sums on GPU
			s=numBlocks;	

			while( s > cpuFinalThreshold) 
			{
				int threads = 0, blocks = 0;		
				oReduce.getNumBlocksAndThreads( s, MAX_THREADS, blocks, threads);

				oReduce.reduce6_max( s, threads, blocks, &d_oMax[( j*NO_OF_BANDS + i)*numBlocks], &d_oMax[( j*NO_OF_BANDS + i)*numBlocks]);

				s =( s +( threads*2-1)) /( threads*2);
			}			
		}
	}

	Static::NormNLKernel<<< dimGrid, dimBlock, 0 >>>( in, d_oMax, d_oMin, _im_size, numBlocks) ;	
	CUDA_CHECK( cudaThreadSynchronize());

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			numBlocks = 0;
			numThreads = 0;

			oReduce.getNumBlocksAndThreads( _im_size.w*_im_size.h, MAX_THREADS, numBlocks, numThreads);

			oReduce.reduce6_sum( _im_size.w*_im_size.h, numThreads, numBlocks, &in[( j*NO_OF_BANDS + i)* _im_size.w*_im_size.h], &d_oSum[( j*NO_OF_BANDS + i)*numBlocks]);

			// sum partial block sums on GPU
			s=numBlocks;	

			while( s > cpuFinalThreshold) 
			{
				int threads = 0, blocks = 0;		
				oReduce.getNumBlocksAndThreads( s, MAX_THREADS, blocks, threads);

				oReduce.reduce6_sum( s, threads, blocks, &d_oSum[( j*NO_OF_BANDS + i)*numBlocks], &d_oSum[( j*NO_OF_BANDS + i)*numBlocks]);

				s =( s +( threads*2-1)) /( threads*2);
			}
		}
	}

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			numBlocks = 0;
			numThreads = 0;

			oReduce.getNumBlocksAndThreads( _im_size.w*_im_size.h, MAX_THREADS, numBlocks, numThreads);

			oReduce.reduce6_max( _im_size.w*_im_size.h, numThreads, numBlocks, &in[( j*NO_OF_BANDS + i)* _im_size.w*_im_size.h], &d_oMax[( j*NO_OF_BANDS + i)*numBlocks]);

			// sum partial block sums on GPU
			s=numBlocks;	

			while( s > cpuFinalThreshold) 
			{
				int threads = 0, blocks = 0;		
				oReduce.getNumBlocksAndThreads( s, MAX_THREADS, blocks, threads);

				oReduce.reduce6_max( s, threads, blocks, &d_oMax[( j*NO_OF_BANDS + i)*numBlocks], &d_oMax[( j*NO_OF_BANDS + i)*numBlocks]);

				s =( s +( threads*2-1)) /( threads*2);
			}			
		}
	}

	Static::NormIttiKernel<<< dimGrid, dimBlock, 0 >>>( in, d_oSum, d_oMax, _im_size, numBlocks) ;	
	CUDA_CHECK( cudaThreadSynchronize());


	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)
	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			numBlocks = 0;
			numThreads = 0;

			oReduce.getNumBlocksAndThreads( _im_size.w*_im_size.h, MAX_THREADS, numBlocks, numThreads);

			oReduce.reduce6_max( _im_size.w*_im_size.h, numThreads, numBlocks, &in[( j*NO_OF_BANDS + i)* _im_size.w*_im_size.h], &d_oMax[( j*NO_OF_BANDS + i)*numBlocks]);

			// sum partial block sums on GPU
			s=numBlocks;	

			while( s > cpuFinalThreshold) 
			{
				int threads = 0, blocks = 0;		
				oReduce.getNumBlocksAndThreads( s, MAX_THREADS, blocks, threads);

				oReduce.reduce6_max( s, threads, blocks, &d_oMax[( j*NO_OF_BANDS + i)*numBlocks], &d_oMax[( j*NO_OF_BANDS + i)*numBlocks]);

				s =( s +( threads*2-1)) /( threads*2);
			}
		}
	}

	Static::NormPCFusionKernel<<< dimGrid, dimBlock, dimBlock.x * sizeof( float) >>>( in, d_oMax, _im_size, out, numBlocks) ;	
	CUDA_CHECK( cudaThreadSynchronize());
}

void Static::Normalize::Init() {

	// Reduction
	int numBlocks = 0, numThreads = 0;
	oReduce.getNumBlocksAndThreads( size, MAX_THREADS, numBlocks, numThreads);

	CUDA_CHECK( cudaMalloc(( void**) &d_oSum, NO_OF_ORIENTS*NO_OF_BANDS * numBlocks * sizeof( float)));
	CUDA_CHECK( cudaMalloc(( void**) &d_oMax, NO_OF_ORIENTS*NO_OF_BANDS * numBlocks * sizeof( float)));
	CUDA_CHECK( cudaMalloc(( void**) &d_oMin, NO_OF_ORIENTS*NO_OF_BANDS * numBlocks * sizeof( float)));
}

void Static::Normalize::Clean() {

	CUDA_CHECK( cudaFree( d_oSum));
	CUDA_CHECK( cudaFree( d_oMax));
	CUDA_CHECK( cudaFree( d_oMin));
}