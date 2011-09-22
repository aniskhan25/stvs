
// includes
#include "../inc/error.hpp"
#include "../inc/STA_Gabor.hpp"


/// <summary>Texture memory for rotations</summary>
texture<float, 2, cudaReadModeElementType> texGaborU;
texture<float, 2, cudaReadModeElementType> texGaborV;


/// <summary>Constant memory for orientations and frequencies</summary>
__constant__ float d_frequencies[NO_OF_BANDS];
__constant__ float d_sig_hor[NO_OF_BANDS];



/// <summary>Initializes the 2D Gabor bank.</summary>
/// <remarks>
/// The function configures the 2D gabor filter bank for several orientations and frequencies.
/// Step 1: Initialize the orientations.
/// Step 2: Initialize the frequencies.
/// Step 3: Compute the gabor functions.
/// Step 4: Bind all the above to GPU memory.
/// </remarks>
void Static::Gabor::Init()
{
	/// <summary>Step 1: Initialize the orientations.</summary>
	for( int i = 0;i < NO_OF_ORIENTS;i++) {

		h_teta[i] = i * 180.0 / NO_OF_ORIENTS + 90.0;
	}

	/// <summary>Step 2: Initialize the frequencies.</summary>
	int index = 0;
	for( int i = NO_OF_BANDS-1;i > -1;i--) {

		h_frequencies[index] = FREQ_MAX /( pow( SCALE, i) * 1.0f);
		h_sig_hor[index]	 =  STD_MAX /( pow( SCALE, i)		);

		index++;
	}

	/// <summary>Step 3: Compute the gabor functions.</summary>
	h_gaborMaskU =( float *)malloc( size_*NO_OF_ORIENTS * sizeof( float));
	h_gaborMaskV =( float *)malloc( size_*NO_OF_ORIENTS * sizeof( float));

	if( h_gaborMaskU == NULL || h_gaborMaskV == NULL) return;

	Static::Gabor::CreateGaborMasks( h_gaborMaskU, h_gaborMaskV, im_size_, h_teta);

	/// <summary>Step 4: Bind all the above to GPU memory.</summary>
	CUDA_CHECK( cudaMemcpyToSymbol( d_frequencies, h_frequencies, NO_OF_BANDS*sizeof( float)));
	CUDA_CHECK( cudaMemcpyToSymbol( d_sig_hor		, h_sig_hor	, NO_OF_BANDS*sizeof( float)));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat);

	CUDA_CHECK( cudaMallocArray( &cu_gaborU, &channelDesc, im_size_.w, NO_OF_ORIENTS*im_size_.h));
	CUDA_CHECK( cudaMemcpyToArray( cu_gaborU, 0, 0, h_gaborMaskU, NO_OF_ORIENTS*size_ * sizeof( float), cudaMemcpyHostToDevice));

	CUDA_CHECK( cudaMallocArray( &cu_gaborV, &channelDesc, im_size_.w, NO_OF_ORIENTS*im_size_.h));
	CUDA_CHECK( cudaMemcpyToArray( cu_gaborV, 0, 0, h_gaborMaskV, NO_OF_ORIENTS*size_ * sizeof( float), cudaMemcpyHostToDevice));

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


/// <summary>Applies Gabor filter.</summary>
/// <param name="out">Gabor maps.</param>
/// <param name="in">Source image.</param>
/// <param name="im_size">Source image size.</param>
/// <returns>Returns Gabor filtered maps</returns>
/// <remarks>
/// The function returns N gabor filtered results for the input image in different orientations and frequencies.
/// N: NO_OF_ORIENTATIONS * NO_OF_FREQUENCY_BANDS
/// \f$G(x,y) = exp \left \{ -\frac{(x'-f)^2}{2 \sigma^2} + \frac{y'^2}{2 \sigma^2} \right \}\f$
/// </remarks>
void Static::Gabor::Apply(
						  complex_t *out
						  , complex_t *in
						  , siz_t im_size						  
						  )
{
	dim3 dimBlock( 32, 8, 1);	
	dim3 dimGrid( 
		( 2*im_size.w / dimBlock.x) + 1*( 2*im_size.w%dimBlock.x!=0)
		, ( im_size.h / dimBlock.y) + 1*( im_size.h%dimBlock.y!=0)
		, 1
		);

	Static::KernelGabor<<< dimGrid, dimBlock, 0 >>>( out, in, im_size);	
	CUDA_CHECK( cudaDeviceSynchronize());
}


/// <summary>Generates equally spaced vector.</summary>
/// <param name="in">Linear data vector.</param>
/// <param name="d1">Lower limit.</param>
/// <param name="d2">Upper limit.</param>
/// <param name="n">Number of points.</param>
/// <returns>Returns equally spaced vector.</returns>
void Static::Gabor::CreateLinear( 
								 float *vec
								 , float d1
								 , float d2
								 , unsigned int n
								 )
{   
	float diff = d2 - d1;

	int i;
	for( i = 0;i < n - 1;i++) {
		vec[i] = i*diff /( floor( ( float)n) - 1) + d1;
	}

	vec[i] = d2;
}


/// <summary>Computes gabor filter masks.</summary>
/// <param name="u1">Rotation x'.</param>
/// <param name="v1">Rotation y'.</param>
/// <param name="teta">Orientation.</param>
/// <param name="im_size">Image size.</param>
/// <returns>Returns the rotations x and y.</returns>
/// <remarks>
/// \f${x}' = xCos\theta + ySin\theta\f$
/// \f${y}' = -xSin\theta + yCos\theta\f$
/// </remarks>
void Static::Gabor::CreateGaborMasks( 
									 float *u1
									 , float *v1									 
									 , siz_t im_size
									 , double *teta
									 )
{
	double theta;

	float *u =( float *) malloc( sizeof( float) *( im_size.w + 1));
	float *v =( float *) malloc( sizeof( float) *( im_size.h + 1));

	if( u == NULL || v == NULL) return;

	CreateLinear( u, -0.5f, 0.5f, im_size.w + 1);
	CreateLinear( v, 0.5f, -0.5f, im_size.h + 1);

	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++) {
		theta = teta[j] * _PI / 180.0;

		for( unsigned int y = 0;y < im_size.h;y++) {
			for( unsigned int x = 0;x < im_size.w;x++) {
				u1[j *( im_size.h*im_size.w) +( y*im_size.w + x)] = u[x]*cos( theta) + v[y]*sin( theta);
				v1[j *( im_size.h*im_size.w) +( y*im_size.w + x)] = v[y]*cos( theta) - u[x]*sin( theta);				
			}
		}	
	}

	free( u);
	free( v);
}


/// <summary>Cleans up the 2D Gabor bank.</summary>
void Static::Gabor::Clean()
{
	CUDA_CHECK( cudaUnbindTexture( texGaborU));
	cu_gaborU = NULL; CUDA_CHECK( cudaFreeArray( cu_gaborU));

	CUDA_CHECK( cudaUnbindTexture( texGaborV));
	cu_gaborV = NULL; CUDA_CHECK( cudaFreeArray( cu_gaborV));

	free( h_gaborMaskU);
	free( h_gaborMaskV);
}


/// <summary>GPU kernel applying Gabor filter.</summary>
/// <param name="in">Source image.</param>
/// <param name="im_size">Source image size.</param>
/// <param name="maps">Gabor maps.</param>
/// <returns>Returns Gabor filtered maps</returns>
__global__  void Static::KernelGabor(
									 complex_t* out
									 , complex_t* in
									 , siz_t im_size									 
									 ) 
{
	unsigned int x = blockIdx.x*blockDim.x / 2 + threadIdx.x/2;
	unsigned int y = blockIdx.y*blockDim.y   + threadIdx.y;

	if( x >= im_size.w || y >= im_size.h) return;

	unsigned int mod = threadIdx.x % 2;

	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++)	{
		for( unsigned int i = 0;i < NO_OF_BANDS;i++) {
			out[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)][mod] = 
				in[y*im_size.w + x][mod] *( 
				__expf( -( 
				( ( tex2D( texGaborU, x, j*im_size.h + y) - d_frequencies[i]) *( tex2D( texGaborU, x, j*im_size.h + y) - d_frequencies[i])
				/( 2.0f *( d_sig_hor[i]*d_sig_hor[i]))) +
				( tex2D( texGaborV, x, j*im_size.h + y) * tex2D( texGaborV, x, j*im_size.h + y)
				/( 2.0f *( d_sig_hor[i]*d_sig_hor[i])))
				))
				);
		}
	}
}