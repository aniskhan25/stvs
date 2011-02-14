#include "error.hpp"

#include "STA_Gabor.hpp"

texture<float, 2, cudaReadModeElementType> texGaborU;
texture<float, 2, cudaReadModeElementType> texGaborV;

__constant__ float d_frequencies[NO_OF_BANDS];
__constant__ float d_sig_hor[NO_OF_BANDS];

void Static::Gabor::CreateLinear( float *in, float d1, float d2, unsigned int n)
{   
	float diff = d2 - d1;

	int i;

	for( i = 0; i < n - 1 ; i++)
		in[i] = i*diff /( floor(( float)n) - 1) + d1;

	in[i] = d2;
}

/**
 Create gabor masks
*/
void Static::Gabor::CreateGaborMasks( float *u1, float *v1, double *teta, siz_t im_size)
{
	double theta;

	float *u =( float *) malloc( sizeof( float) *( im_size.w + 1));
	float *v =( float *) malloc( sizeof( float) *( im_size.h + 1));

	if( u == NULL || v == NULL) return;

	CreateLinear( u, -0.5f, 0.5f, im_size.w + 1);
	CreateLinear( v, 0.5f, -0.5f, im_size.h + 1);

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)	
	{
		theta = teta[j] * PI / 180.0;
		
		for( unsigned int y = 0 ; y < im_size.h ; y++) {
			for( unsigned int x = 0 ; x < im_size.w ; x++) {

				u1[j *( im_size.h*im_size.w) +( y*im_size.w + x)] = u[x]*cos( theta) + v[y]*sin( theta);
				v1[j *( im_size.h*im_size.w) +( y*im_size.w + x)] = v[y]*cos( theta) - u[x]*sin( theta);				
			}
		}	
	}

	free( u);
	free( v);
}

/**
 Gabor bank kernel
*/
__global__  void Static::GaborKernel( complex_t* in, siz_t im_size, complex_t* maps) 
{
	unsigned int x = blockIdx.x*blockDim.x / 2 + threadIdx.x/2;
	unsigned int y = blockIdx.y*blockDim.y   + threadIdx.y;

	if( x >= im_size.w || y >= im_size.h) return;

	unsigned int mod = threadIdx.x % 2;

	for( unsigned int j = 0; j < NO_OF_ORIENTS ; j++)	{
		for( unsigned int i = 0 ; i < NO_OF_BANDS ; i++) {

			maps[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)][mod] = 
				in[y*im_size.w + x][mod] *( 
				__expf( -( 
				(( tex2D( texGaborU, x, j*im_size.h + y) - d_frequencies[i]) *( tex2D( texGaborU, x, j*im_size.h + y) - d_frequencies[i])
				/( 2.0f *( d_sig_hor[i]*d_sig_hor[i]))) +
				( tex2D( texGaborV, x, j*im_size.h + y) * tex2D( texGaborV, x, j*im_size.h + y)
				/( 2.0f *( d_sig_hor[i]*d_sig_hor[i])))
				))
				);
		}
	}
}

/**
 Invoke gabor bank kernel
*/
void Static::Gabor::Apply( complex_t *in, siz_t im_size, complex_t *out)
{
	dim3 dimBlock( 32, 8, 1);	
	dim3 dimGrid(( 2*im_size.w / dimBlock.x) + 1*( 2*im_size.w%dimBlock.x!=0)
		,( im_size.h / dimBlock.y) + 1*( im_size.h%dimBlock.y!=0), 1);

	Static::GaborKernel<<< dimGrid, dimBlock, 0 >>>( in, im_size, out);	
	CUDA_CHECK( cudaThreadSynchronize());
}

void Static::Gabor::Init()
{
	/**
	Initialize orientations and frequency bands
	*/
	for( int i = 0 ; i < NO_OF_ORIENTS ; i++) {

		h_teta[i] = i * 180.0 / NO_OF_ORIENTS + 90.0;
	}

	int index = 0;
	for( int i = NO_OF_BANDS-1 ; i > -1 ; i--) {

		h_frequencies[index] = FREQ_MAX /( pow( SCALE, i) * 1.0f);
		h_sig_hor[index]	 =  STD_MAX /( pow( SCALE, i)		);

		index++;
	}

	CUDA_CHECK( cudaMemcpyToSymbol( d_frequencies, h_frequencies, NO_OF_BANDS*sizeof( float)));
	CUDA_CHECK( cudaMemcpyToSymbol( d_sig_hor		, h_sig_hor	, NO_OF_BANDS*sizeof( float)));

	/**
	Create gabor masks
	*/
	h_gaborMaskU =( float *)malloc( size*NO_OF_ORIENTS * sizeof( float));
	h_gaborMaskV =( float *)malloc( size*NO_OF_ORIENTS * sizeof( float));

	if( h_gaborMaskU == NULL || h_gaborMaskV == NULL) return;

	Static::Gabor::CreateGaborMasks( h_gaborMaskU, h_gaborMaskV, h_teta, _im_size);

	/**
	Bind the mask to texture memory
	*/
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat);

	CUDA_CHECK( cudaMallocArray( &cu_gaborU, &channelDesc, _im_size.w, NO_OF_ORIENTS*_im_size.h)); 
	CUDA_CHECK( cudaMemcpyToArray( cu_gaborU, 0, 0, h_gaborMaskU, NO_OF_ORIENTS*size * sizeof( float), cudaMemcpyHostToDevice));

	CUDA_CHECK( cudaMallocArray( &cu_gaborV, &channelDesc, _im_size.w, NO_OF_ORIENTS*_im_size.h)); 
	CUDA_CHECK( cudaMemcpyToArray( cu_gaborV, 0, 0, h_gaborMaskV, NO_OF_ORIENTS*size * sizeof( float), cudaMemcpyHostToDevice));

	texGaborU.addressMode[0] = cudaAddressModeClamp;
	texGaborU.addressMode[1] = cudaAddressModeClamp;
	texGaborU.filterMode	 = cudaFilterModePoint;
	texGaborU.normalized	 = false;

	texGaborV.addressMode[0] = cudaAddressModeClamp; 
	texGaborV.addressMode[1] = cudaAddressModeClamp;
	texGaborV.filterMode	 = cudaFilterModePoint;
	texGaborV.normalized	 = false;

	CUDA_CHECK( cudaBindTextureToArray( texGaborU, cu_gaborU, channelDesc));
	CUDA_CHECK( cudaBindTextureToArray( texGaborV, cu_gaborV, channelDesc));
}

/**
 Cleanup
*/void Static::Gabor::Clean()
{
	CUDA_CHECK( cudaUnbindTexture( texGaborU));
	CUDA_CHECK( cudaFreeArray( cu_gaborU));

	CUDA_CHECK( cudaUnbindTexture( texGaborV));
	CUDA_CHECK( cudaFreeArray( cu_gaborV));

	free( h_gaborMaskU);
	free( h_gaborMaskV);
}