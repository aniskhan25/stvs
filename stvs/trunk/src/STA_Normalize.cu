
// includes
//#include <shrUtils.h>
//#include <shrQATest.h>

#include "../inc/error.hpp"
#include "../inc/STA_Normalize.hpp"

//#define VERBOSE

/// <summary>Initializes the mask.</summary>
void Static::Normalize::Init() 
{
	result = ( summary_stats_data*) malloc( NO_OF_ORIENTS*NO_OF_BANDS * sizeof( summary_stats_data));

	CUDA_CHECK( cudaMalloc( ( void**)&d_result, NO_OF_ORIENTS*NO_OF_BANDS * sizeof( summary_stats_data)));
}


/// <summary>Normalizes the feature maps and fuses them.</summary>
/// <param name="in">Feature maps.</param>
/// <param name="im_size">Image size.</param>
/// <param name="out">Destination image.</param>
/// <returns>Returns the static saliency map.</returns>
/// <remarks>
/// The function normalizes the feature maps using several methods before summing them up into the static saliency map.
/// Step 1: Normalization.
/// Step 2: Normalization Itti.
/// Step 3: Normalization and fusion.
/// </remarks>
void Static::Normalize::Apply( 
							  float *out
							  , float* in
							  , siz_t im_size							  
							  )
{
	dim3 dimBlock( 64, 1, 1);	
	dim3 dimGrid( 
		( im_size.w / dimBlock.x) + 1*( im_size.w%dimBlock.x!=0)
		, ( im_size.h / dimBlock.y) + 1*( im_size.h%dimBlock.y!=0)
		, dimBlock.z
		);

		#ifdef VERBOSE
			shrLog("red1\n");
#endif

	/// <summary>
	/// Step 1: Finds the local minimum and maximum of the individual feature maps that are used by the normalization.
	/// </summary>	
	for( unsigned int i = 0;i < NO_OF_ORIENTS*NO_OF_BANDS;i++){
		result[i] = oReduce.Apply(&in[i * im_size.w*im_size.h], im_size);
	}	

	CUDA_CHECK( cudaMemcpy( d_result, result, NO_OF_ORIENTS*NO_OF_BANDS * sizeof( summary_stats_data), cudaMemcpyHostToDevice));

		#ifdef VERBOSE
			shrLog("norm1\n");
#endif

	Static::KernelNormalizeNL<<< dimGrid, dimBlock, 0 >>>( in, d_result, im_size);	
	CUDA_CHECK( cudaDeviceSynchronize());

		#ifdef VERBOSE
			shrLog("red2\n");
#endif

	/// <summary>
	/// Step 2: Finds the local sum and maximum of the individual feature maps that are used by the normalization.
	/// </summary>
	for( unsigned int i = 0;i < NO_OF_ORIENTS*NO_OF_BANDS;i++){
		result[i] = oReduce.Apply(&in[i * im_size.w*im_size.h], im_size);
	}	

	CUDA_CHECK( cudaMemcpy( d_result, result, NO_OF_ORIENTS*NO_OF_BANDS * sizeof( summary_stats_data), cudaMemcpyHostToDevice));

		#ifdef VERBOSE
			shrLog("norm2\n");
#endif

	Static::KernelNormalizeItti<<< dimGrid, dimBlock, 0 >>>( in, d_result, im_size);	
	CUDA_CHECK( cudaDeviceSynchronize());

		#ifdef VERBOSE
			shrLog("red3\n");
#endif

	/// <summary>
	/// Step 3: Finds the local maximum of the individual feature maps that are used by the normalization, and finally resulting in the saliency map.
	/// </summary>
	for( unsigned int i = 0;i < NO_OF_ORIENTS*NO_OF_BANDS;i++){
		result[i] = oReduce.Apply(&in[i * im_size.w*im_size.h], im_size);
	}	

	CUDA_CHECK( cudaMemcpy( d_result, result, NO_OF_ORIENTS*NO_OF_BANDS * sizeof( summary_stats_data), cudaMemcpyHostToDevice));

		#ifdef VERBOSE
			shrLog("norm3\n");
#endif

	Static::KernelNormalizePCFusion<<< dimGrid, dimBlock, dimBlock.x * sizeof( float) >>>( out, in, d_result, im_size);	
	CUDA_CHECK( cudaDeviceSynchronize());
}


/// <summary>Cleans up the mask.</summary>
void Static::Normalize::Clean() 
{
	free(result);

	d_result = NULL; CUDA_CHECK( cudaFree( d_result));
}


/// <summary>GPU kernel to normalizes the feature maps.</summary>
/// <param name="maps">Feature maps.</param>
/// <param name="maxpt">Local Maximum of feature maps.</param>
/// <param name="minpt">Local Minimum of feature maps.</param>
/// <param name="im_size">Image size.</param>
/// <param name="blocks">No. of blocks.</param>
/// <returns>Returns in normalized feature maps.</returns>
/// <remarks>
/// The function normalizes the feature maps using:
/// \frac{\left ( x - \underset{x}{\operatorname{argmin}} \right )}{\underset{x}{\operatorname{argmax}} - \underset{x}{\operatorname{argmin}}}
/// </remarks>
__global__  void Static::KernelNormalizeNL( 
	float* maps
	, summary_stats_data* result
	, siz_t im_size	
	)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float maximum, minimum, tmp;

	if( x>=im_size.w || y>=im_size.h) return;

	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++){
		for( unsigned int i = 0;i < NO_OF_BANDS;i++){
			maximum = result[i + j*NO_OF_BANDS].max;
			minimum = result[i + j*NO_OF_BANDS].min;

			tmp  = maps[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)];

			// Normalize
			if( minimum != maximum){
				tmp  -= minimum;
				tmp /=( maximum - minimum);
			}	

			maps[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)] = tmp;
		}
	}
}


/// <summary>GPU kernel to applies Itti's normalization to the feature maps.</summary>
/// <param name="maps">Feature maps.</param>
/// <param name="maxpt">Local Maximum of feature maps.</param>
/// <param name="sumpt">Local Sum of feature maps.</param>
/// <param name="im_size">Image size.</param>
/// <param name="blocks">No. of blocks.</param>
/// <returns>Returns in normalized feature maps.</returns>
/// <remarks>
/// The function normalizes the feature maps using the Itti's normalization:
/// \left ( \underset{x}{argmax}f(x)-\bar{x}  \right )^2 
/// </remarks>
__global__  void Static::KernelNormalizeItti(
	float* maps
	, summary_stats_data* result	
	, siz_t im_size
	)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	float maximum, sum;

	if( x>=im_size.w || y>=im_size.h) return;

	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++){
		for( unsigned int i = 0;i < NO_OF_BANDS;i++){
			maximum = result[i + j*NO_OF_BANDS].max;
			sum		= result[i + j*NO_OF_BANDS].sum;

			// Normalization Itti

			if( sum != maximum)
				maps[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)] = maps[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)]	*( ( maximum - sum /( im_size.w*im_size.h)) *( maximum - sum /( im_size.w*im_size.h)));
		}
	}
}

/// <summary>GPU kernel to normalize and fuse all the feature maps into static saliency map.</summary>
/// <param name="out">Feature maps.</param>
/// <param name="maps">Destination image.</param>
/// <param name="maxpt">Local Maximum of feature maps.</param>
/// <param name="im_size">Image size.</param>
/// <param name="blocks">No. of blocks.</param>
/// <returns>Returns the static saliency map.</returns>
/// <remarks>
/// The function normalizes the feature maps by taking only 20% of the values above the threshold (maximum value of the individual feature map).
/// Then fuses the normalized maps through simple summation into a saliency map for the static pathway of the STVS model.
/// </remarks>
__global__  void Static::KernelNormalizePCFusion(
	float* out
	, float* maps
	, summary_stats_data* result
	, siz_t im_size		
	)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if( x>=im_size.w || y>=im_size.h) return;

	extern __shared__ float buf[];	// based on no. of threads

	float maximum, tmp, level;

	buf[threadIdx.x] = 0.0f;

	__syncthreads();

	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++){
		for( unsigned int i = 0;i < NO_OF_BANDS;i++){
			maximum = result[i + j*NO_OF_BANDS].max;

			tmp		= maps[( j*NO_OF_BANDS + i) *( im_size.w*im_size.h) +( y*im_size.w + x)];

			level = 0.2f * maximum;

			if( tmp <= level)
				tmp = 0.0f;
			else
				tmp =( tmp - level) /( maximum - level) * maximum;

			buf[threadIdx.x] += tmp;			

			__syncthreads();
		}
	}

	out[y*im_size.w + x] = buf[threadIdx.x];	
}