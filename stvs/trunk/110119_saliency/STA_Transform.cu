#include "cufft_common.hpp"

#include "STA_Transform.hpp"

/**
Call to CuFFT library
*/
void Static::Transform::FFT( complex_t *idata, siz_t im_size, complex_t* odata, int direction)
{
	cufftHandle plan;

	CUFFT_CHECK( cufftPlan2d( &plan, im_size.h, im_size.w, CUFFT_C2C));

	CUFFT_CHECK( cufftExecC2C( plan,( cufftComplex*)idata,( cufftComplex*)odata, direction));

	CUFFT_CHECK( cufftDestroy( plan));		
}

/**
Shift kernel
*/
__global__  void Static::FFTShiftKernel( complex_t* in, siz_t _im_size, unsigned int centerW, unsigned int centerH, unsigned int widthEO, unsigned int heightEO, complex_t* out) 
{
	unsigned int x	= blockIdx.x*blockDim.x / 2 + threadIdx.x / 2;
	unsigned int y	= blockIdx.y*blockDim.y		+ threadIdx.y;

	if( x>=_im_size.w || y>=_im_size.h) return;

	unsigned int xx	=( x < centerW) ?( x+centerW + widthEO) :( x - centerW);
	unsigned int yy	=( y < centerH) ?( y+centerH + heightEO) :( y - centerH);

	unsigned int mod	= threadIdx.x % 2;

	out[yy*_im_size.w + xx][mod] = in[y*_im_size.w + x][mod];
}

/**
Inverse shift kernel
*/
__global__  void Static::IFFTShiftKernel( complex_t* inMaps, siz_t _im_size, unsigned int centerW, unsigned int centerH, unsigned int widthEO, unsigned int heightEO, complex_t* outMaps) 
{
	unsigned int x	 = blockIdx.x*blockDim.x / 2 + threadIdx.x/2;
	unsigned int y	 = blockIdx.y*blockDim.y	 + threadIdx.y;

	if( x>=_im_size.w || y>=_im_size.h) return;

	unsigned int mod = threadIdx.x % 2;

	unsigned int xx =( x < centerW) ?( x+centerW + widthEO) :( x-centerW);
	unsigned int yy =( y < centerH) ?( y+centerH + heightEO) :( y-centerH);

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++) {
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++) {

			outMaps[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( y*_im_size.w + x)][mod] = inMaps[( j*NO_OF_BANDS + i) *( _im_size.w*_im_size.h) +( yy*_im_size.w + xx)][mod];
		}
	}
}

void Static::Transform::Apply( complex_t* in, siz_t _im_size, complex_t* out, int direction)
{
	dim3 dimBlock( 32, 8, 1);	
	dim3 dimGrid(( 2*_im_size.w / dimBlock.x) + 1*( 2*_im_size.w%dimBlock.x!=0),( _im_size.h / dimBlock.y) + 1*( _im_size.h%dimBlock.y!=0), dimBlock.z);

	/**
	FFT + Shift function
	*/
	switch( direction) {
	case CUFFT_FORWARD :
		FFT( in, _im_size, in, CUFFT_FORWARD);

		Static::FFTShiftKernel<<< dimGrid, dimBlock, 0 >>>( in, _im_size, _im_size.w/2, _im_size.h/2,(( _im_size.w%2 == 0) ? 0 : 1),(( _im_size.h%2 == 0) ? 0 : 1), out);		
		CUDA_CHECK( cudaThreadSynchronize());
		break;

		/**
		Inverse shift function + IFFT
		*/
	case CUFFT_INVERSE :

		Static::IFFTShiftKernel<<< dimGrid, dimBlock, 0 >>>( in, _im_size, _im_size.w/2, _im_size.h/2,(( _im_size.w%2 == 0) ? 0 : 1),(( _im_size.h%2 == 0) ? 0 : 1), out);		
		CUDA_CHECK( cudaThreadSynchronize());

		for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)	{
			for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++) {

				FFT( &out[( j*NO_OF_BANDS + i)*( _im_size.w*_im_size.h)]
				, _im_size, &out[( j*NO_OF_BANDS + i)*( _im_size.w*_im_size.h)]
				, CUFFT_INVERSE);
			}
		}
		break;

	}
}

// TODO
void Static::Transform::Init(){}
void Static::Transform::Clean(){}