#include "error.hpp"

#include "STA_Mask.hpp"

texture<float, 2, cudaReadModeElementType> texMask;

void Static::Mask::Apply( float *in, siz_t const & _im_size, float *out) 
{
	dim3 dimBlock( 8, 8, 1);	
	dim3 dimGrid(( _im_size.w / dimBlock.x) + 1*( _im_size.w%dimBlock.x!=0),( _im_size.h / dimBlock.y) + 1*( _im_size.h%dimBlock.y!=0), dimBlock.z);

	Static::MaskKernel<<< dimGrid, dimBlock, 0 >>>( in, _im_size, out);
	CUDA_CHECK( cudaThreadSynchronize());
}

void Static::Mask::Apply( float *in, siz_t const & _im_size, complex_t *out) 
{
	dim3 dimBlock( 8, 8, 1);	
	dim3 dimGrid(( _im_size.w / dimBlock.x) + 1*( _im_size.w%dimBlock.x!=0),( _im_size.h / dimBlock.y) + 1*( _im_size.h%dimBlock.y!=0), dimBlock.z);

	Static::MaskKernel<<< dimGrid, dimBlock, 0 >>>( in, _im_size, out);
	CUDA_CHECK( cudaThreadSynchronize());
}

__global__  void Static::MaskKernel( float* in, siz_t _im_size, float *out) 
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=_im_size.w || y>=_im_size.h) return;

	float u = x /( float) _im_size.w;
	float v = y /( float) _im_size.h;

	out[y*_im_size.w + x] = in[y*_im_size.w + x] * tex2D( texMask, u, v);	
}

/**
 Mask kernels
*/
__global__  void Static::MaskKernel( float* in, siz_t _im_size, complex_t* out) 
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=_im_size.w || y>=_im_size.h) return;

	float u = x /( float) _im_size.w;
	float v = y /( float) _im_size.h;

	out[y*_im_size.w + x][0] = in[y*_im_size.w + x] * tex2D( texMask, u, v);
	out[y*_im_size.w + x][1] = 0.0f;	
}

void Static::Mask::Init()
{    
	/**
	Create hanning mask
	*/
	h_mask =( float *)malloc( size * sizeof( float));
	Static::Mask::CreateMask( h_mask, _im_size.h, _im_size.w, 8);

	/**
	Bind mask to texture memory
	*/
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat);

	CUDA_CHECK( cudaMallocArray( &cu_mask, &channelDesc, _im_size.w, _im_size.h)); 
	CUDA_CHECK( cudaMemcpyToArray( cu_mask, 0, 0, h_mask, size * sizeof( float), cudaMemcpyHostToDevice));

	texMask.addressMode[0] = cudaAddressModeClamp;
	texMask.addressMode[1] = cudaAddressModeClamp;
	texMask.filterMode = cudaFilterModePoint;
	texMask.normalized = true;

	CUDA_CHECK( cudaBindTextureToArray( texMask, cu_mask, channelDesc));
}

/**
Cleanup
*/
void Static::Mask::Clean() 
{
	CUDA_CHECK( cudaUnbindTexture( texMask));

	CUDA_CHECK( cudaFreeArray( cu_mask));

	free( h_mask);
}

/**
Create hanning mask
*/
int Static::Mask::CreateMask( float *mask, unsigned int rows, unsigned int cols, int mu)
{    
	float var1 =( float)floor( cols / 2.0f);
	float var2 = pow( var1, mu);

	float var3 =( float)floor( rows / 2.0f);
	float var4 = pow( var3, mu);

	for( unsigned int i = 0 ; i < rows ; i++) {
		for( unsigned int j = 0 ; j < cols ; j++) {
			mask[i*cols + j] =( 
				( 1.0f -( pow(( j - var1), mu) / var2)) *
				( 1.0f -( pow(( i - var3), mu) / var4))
				);
		}
	}

	return 1;
}