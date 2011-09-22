
// includes
#include "../inc/error.hpp"
#include "../inc/struct.hpp"
#include "../inc/STA_Mask.hpp"


/// <summary>GPU kernel applying Gabor filter.</summary>
texture<float, 2, cudaReadModeElementType> texMask;


/// <summary>Initializes the mask.</summary>
void Static::Mask::Init()
{    
	h_mask =( float *)malloc( size_ * sizeof( float));
	Static::Mask::CreateMask( h_mask, im_size_, 8);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat);

	CUDA_CHECK( cudaMallocArray( &cu_mask, &channelDesc, im_size_.w, im_size_.h));
	CUDA_CHECK( cudaMemcpyToArray( cu_mask, 0, 0, h_mask, size_ * sizeof( float), cudaMemcpyHostToDevice));

	texMask.addressMode[0] = cudaAddressModeClamp;
	texMask.addressMode[1] = cudaAddressModeClamp;
	texMask.filterMode = cudaFilterModePoint;
	texMask.normalized = true;

	CUDA_CHECK( cudaBindTextureToArray( texMask, cu_mask, channelDesc));
}


/// <summary>Computes the mask.</summary>
/// <param name="mask">Output mask.</param>
/// <param name="im_size">Mask size.</param>
/// <param name="mu">Sigma</param>
/// <returns>Returns the mask.</returns>
/// <remarks>
/// The function creates a mask similar to a hanning function to discard the unnecessary boundaries.
/// </remarks>
void Static::Mask::CreateMask( 
							 float *mask
							 , siz_t im_size
							 , int mu
							 )
{    
	float var1 =( float)floor( im_size.w/2.0f);
	float var2 = pow( var1, mu);

	float var3 =( float)floor( im_size.h/2.0f);
	float var4 = pow( var3, mu);

	for( unsigned int i = 0;i < im_size.h;i++) {
		for( unsigned int j = 0;j < im_size.w;j++) {
			mask[i*im_size.w + j] =( 
				( 1.0f -( pow( ( j - var1), mu) / var2)) *
				( 1.0f -( pow( ( i - var3), mu) / var4))
				);
		}
	}
}


/// <summary>Applies the mask.</summary>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="out">Destination image.</param>
/// <returns>Returns the masked image</returns>
void Static::Mask::Apply(
						 float *out
						 , float *in
						 , siz_t const & im_size						 
						 ) 
{
	dim3 dimBlock( 8, 8, 1);	
	dim3 dimGrid( 
		( im_size.w / dimBlock.x) + 1*( im_size.w%dimBlock.x!=0)
		, ( im_size_.h / dimBlock.y) + 1*( im_size.h%dimBlock.y!=0)
		, dimBlock.z
		);

	Static::MaskKernel<<< dimGrid, dimBlock, 0 >>>( out, in, im_size);
	CUDA_CHECK( cudaDeviceSynchronize());
}


/// <summary>Applies the mask.</summary>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="out">Destination image.</param>
/// <returns>Returns the masked image</returns>
void Static::Mask::Apply(
						 complex_t *out
						 , float *in
						 , siz_t const & im_size						 
						 ) 
{
	dim3 dimBlock( 8, 8, 1);	
	dim3 dimGrid( 
		( im_size.w / dimBlock.x) + 1*( im_size.w%dimBlock.x!=0)
		, ( im_size.h / dimBlock.y) + 1*( im_size.h%dimBlock.y!=0)
		, dimBlock.z
		);

	Static::MaskKernel<<< dimGrid, dimBlock, 0 >>>( out, in, im_size);
	CUDA_CHECK( cudaDeviceSynchronize());
}


/// <summary>Cleans up the mask.</summary>
void Static::Mask::Clean() 
{
	CUDA_CHECK( cudaUnbindTexture( texMask));
	cu_mask = NULL; CUDA_CHECK( cudaFreeArray( cu_mask));

	free( h_mask);
}


/// <summary>GPU kernel to apply the mask.</summary>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="out">Destination image.</param>
/// <returns>Returns the masked image</returns>
__global__  void Static::MaskKernel(
									float *out
									, float* in
									, siz_t im_size									
									) 
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=im_size.w || y>=im_size.h) return;

	float u = x /( float) im_size.w;
	float v = y /( float) im_size.h;

	out[y*im_size.w + x] = in[y*im_size.w + x] * tex2D( texMask, u, v);	
}


/// <summary>GPU kernel to apply the mask.</summary>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <param name="out">Destination image.</param>
/// <returns>Returns the masked image</returns>
__global__  void Static::MaskKernel(
									complex_t* out
									, float* in
									, siz_t im_size									
									) 
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=im_size.w || y>=im_size.h) return;

	float u = x /( float) im_size.w;
	float v = y /( float) im_size.h;

	out[y*im_size.w + x][0] = in[y*im_size.w + x] * tex2D( texMask, u, v);
	out[y*im_size.w + x][1] = 0.0f;	
}