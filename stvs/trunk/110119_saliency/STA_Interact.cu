#include "error.hpp"

#include "STA_Interact.hpp"

/**
 Interaction Kernel
*/
__global__  void Static::ShortInteractionKernel( complex_t* in, siz_t _im_size, float* mapsout) 
{
	unsigned int width = _im_size.w;
	unsigned int height = _im_size.h;

	unsigned int x   = blockIdx.x*blockDim.x + threadIdx.x/2;
	unsigned int x2  = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y   = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=width || y>=height) return;

	unsigned int mod = threadIdx.x % 2;

	__shared__ float maps[NO_OF_ORIENTS*NO_OF_BANDS][32];
	__shared__ float buf [72];

	unsigned int i, j;
	unsigned pt = threadIdx.x/2	+ 40*mod	;

	//convert and prefetch
	for( j = 0; j < NO_OF_ORIENTS ; ++j)	
	{
		for( i = 0 ; i < NO_OF_BANDS ; ++i)		
		{   
			//for coalesced memory access: 32 threads process 16 complex in parallel and store them with real and imaginary interlaced
			buf[pt	 ]	= in[( j*NO_OF_BANDS + i) *( width*height) +( y*width + x	)][mod] /( float)( width*height);
			//for coalesced memory access: 32 threads process 16 next complex in parallel
			buf[pt+16]	= in[( j*NO_OF_BANDS + i) *( width*height) +( y*width + x + 16)][mod] /( float)( width*height);
			__syncthreads();

			//for coalesced memory access: 32 threads produce 32 real products in parallel
			maps[j*NO_OF_BANDS + i][threadIdx.x] = abs( buf[threadIdx.x]*buf[threadIdx.x] + buf[40 + threadIdx.x]*buf[40 + threadIdx.x]);
		} 
	}

	__syncthreads();

	unsigned int jp, jm;
	float k1, k2, k3;

	for( j = 0; j < NO_OF_ORIENTS ; j++)	
	{
		for( i = 0 ; i < NO_OF_BANDS ; i++)		
		{
			jp = j + 1;
			jm = j - 1;

			if( j == NO_OF_ORIENTS-1) jp = 0;
			if( j == 0				) jm = NO_OF_ORIENTS-1;

			k1 = 0.5f; k2 = 0.5f; k3 = 0.5f;

			if( i == 0			) {k2 = 0.0f; k3 = 0.25f;}
			if( i == NO_OF_BANDS-1) {k1 = 0.0f; k3 = 0.25f;}

			mapsout[( j*NO_OF_BANDS + i) *( width*height) +( y*width + x2)] =
				maps[ j*NO_OF_BANDS + i    ][threadIdx.x] +
				k2 * maps[ j*NO_OF_BANDS + i - 1][threadIdx.x] +
				k1 * maps[ j*NO_OF_BANDS + i + 1][threadIdx.x] -
				k3 * maps[jp*NO_OF_BANDS + i    ][threadIdx.x] -
				k3 * maps[jm*NO_OF_BANDS + i    ][threadIdx.x];
		}	
	}  
}

void Static::Interact::Apply( complex_t* in, siz_t im_size, float* out) {
	/**
	Interaction
	*/
	dim3 dimBlock( 32, 1, 1);	
	dim3 dimGrid(( im_size.w / dimBlock.x) *( im_size.w%dimBlock.x!=0),( im_size.h / dimBlock.y) + 1*( im_size.h%dimBlock.y!=0), dimBlock.z);

	Static::ShortInteractionKernel<<< dimGrid, dimBlock, 0 >>>( in, im_size, out);	
	CUDA_CHECK( cudaThreadSynchronize());
}

void Static::Interact::Init(){}
void Static::Interact::Clean(){}