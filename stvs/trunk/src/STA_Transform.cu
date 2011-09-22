#include "../inc/cufft_common.hpp"
#include "../inc/STA_Transform.hpp"


/// <summary>Initializes the transform operation.</summary>
void Static::Transform::Init()
{	
	CUFFT_CHECK( cufftPlan2d( &plan, im_size_.h, im_size_.w, CUFFT_C2C));
} 


/// <summary>Calls the cufft library tranforms for an image.</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="direction">Transform direction.</param>
/// <returns>Returns the transformed image.</returns>
void Static::Transform::FFT( 
							complex_t* out
							, complex_t *in
							, siz_t im_size
							, int direction
							)
{	

	
	CUFFT_CHECK( cufftExecC2C( plan, ( cufftComplex*)in, ( cufftComplex*)out, direction));
	
}


/// <summary>Cleans up the the transform operation.</summary>
void Static::Transform::Clean()
{
	CUFFT_CHECK( cufftDestroy( plan));
}


/// <summary>Applies im tranforms on an image.</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="direction">Transform direction.</param>
/// <returns>Returns the transformed image.</returns>
void Static::Transform::Apply(
							  complex_t* out
							  , complex_t* in
							  , siz_t im_size							  
							  , int direction
							  )
{
	dim3 dimBlock( 32, 8, 1);
	dim3 dimGrid( 
		iDivUp( 2*im_size.w, dimBlock.x)
		, iDivUp( im_size.h, dimBlock.y)
		, 1
		);

	point_t center;
	center.x = im_size.w/2;
	center.y = im_size.h/2;

	switch( direction) 
	{
		/// <summary>
		/// FFT + Shift function
		/// </summary>
	case CUFFT_FORWARD:
		FFT( in, in, im_size, CUFFT_FORWARD);

		Static::KernelShift<<< dimGrid, dimBlock, 0 >>>( 
			out
			, in
			, im_size
			, center
			, ( ( im_size.w%2 == 0) ? false : true)
			, ( ( im_size.h%2 == 0) ? false : true)			
			);		
		CUDA_CHECK( cudaDeviceSynchronize());

		break;		

		/// <summary>
		/// Inverse shift function + IFFT
		/// </summary>
	case CUFFT_INVERSE:

		Static::KernelShiftInverse<<< dimGrid, dimBlock, 0 >>>( 
			out
			, in
			, im_size
			, center
			, ( ( im_size.w%2 == 0) ? false : true)
			, ( ( im_size.h%2 == 0) ? false : true)			
			);	
CUDA_CHECK( cudaDeviceSynchronize());
		
		for( unsigned int j = 0;j < NO_OF_ORIENTS;j++){
			for( unsigned int i = 0;i < NO_OF_BANDS;i++){

				FFT( &out[( j*NO_OF_BANDS + i)*( im_size.w*im_size.h)]
				, &out[( j*NO_OF_BANDS + i)*( im_size.w*im_size.h)]
				, im_size					
				, CUFFT_INVERSE);
			}
		}
		break;
	}
}


/// <summary>GPU kernel performing the shift operation on image.</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="center">Center location.</param>
/// <param name="is_width_odd">Is width odd.</param>
/// <param name="is_height_odd">Is height odd.</param>
/// <returns>Returns the shifted image.</returns>
__global__  void Static::KernelShift(
									 complex_t* out
									 , complex_t* in
									 , siz_t im_size
									 , point_t center
									 , bool is_width_odd
									 , bool is_height_odd									 
									 ) 
{
	unsigned int x	= blockIdx.x*blockDim.x / 2 + threadIdx.x / 2;
	unsigned int y	= blockIdx.y*blockDim.y		+ threadIdx.y;

	if( x>=im_size.w || y>=im_size.h) return;

	unsigned int xx	=( x < center.x) ?( x+center.x + is_width_odd ) :( x - center.x);
	unsigned int yy	=( y < center.y) ?( y+center.y + is_height_odd) :( y - center.y);

	unsigned int mod	= threadIdx.x % 2;
	
	out[yy*im_size.w + xx][mod] = in[y*im_size.w + x][mod];
}

/// <summary>GPU kernel performing the shift operation on maps.</summary>
/// <param name="out">Destination maps.</param>
/// <param name="in">Source maps.</param>
/// <param name="im_size">Image size.</param>
/// <param name="center">Center location.</param>
/// <param name="is_width_odd">Is width odd.</param>
/// <param name="is_height_odd">Is height odd.</param>
/// <param name="direction">Transform direction.</param>
/// <returns>Returns the shifted feature maps.</returns>
__global__  void Static::KernelShiftInverse( 
	complex_t* out
	, complex_t* in
	, siz_t im_size
	, point_t center
	, bool is_width_odd
	, bool is_height_odd	
	) 
{
	unsigned int x	 = blockIdx.x*blockDim.x / 2 + threadIdx.x/2;
	unsigned int y	 = blockIdx.y*blockDim.y	 + threadIdx.y;

	if( x>=im_size.w || y>=im_size.h) return;

	unsigned int mod = threadIdx.x % 2;

	unsigned int xx =( x < center.x) ?( x+center.x + is_width_odd ) :( x-center.x);
	unsigned int yy =( y < center.y) ?( y+center.y + is_height_odd) :( y-center.y);

	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++) {
		for( unsigned int i = 0;i < NO_OF_BANDS;i++) {

			out[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)][mod] =
				in[( j*NO_OF_BANDS + i) * ( im_size.w*im_size.h) +( yy*im_size.w + xx)][mod];
		}
	}
}