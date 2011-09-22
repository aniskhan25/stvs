
// includes
#include <assert.h>
#include "cufft.h"
#include "../inc/STA_Pathway.hpp"

//#include <shrUtils.h>
//#include <shrQATest.h>
//
//#define VERBOSE

/// <summary>Property to set the input data onto GPU memory. [float data version]</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
void Static::Pathway::SetData(
							  float* out
							  , float* in
							  ) 
{
	CUDA_CHECK( cudaMemcpy( out, in, size_ * sizeof( float), cudaMemcpyHostToDevice));
}


/// <summary>Property to get the output data from GPU memory.</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
void Static::Pathway::GetData( 
							  float* out
							  , float* in
							  ) 
{
	CUDA_CHECK( cudaMemcpy( out, in, size_*sizeof( float), cudaMemcpyDeviceToHost));
}


/// <summary>Property to get the output data from GPU memory. [Complex data version]</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
void Static::Pathway::GetData( 
							  complex_t* out
							  , complex_t* in
							  )
{
	CUDA_CHECK( cudaMemcpy( out, in, size_*sizeof( complex_t), cudaMemcpyDeviceToHost));
}


/// <summary>Initializes the static pathway of STVS model.</summary>
void Static::Pathway::Init()
{
	CUDA_CHECK( cudaMalloc( ( void**)&d_idata	  , size_ * sizeof( float)));
	CUDA_CHECK( cudaMalloc( ( void**)&d_data_freq , size_ * sizeof( complex_t)));
	CUDA_CHECK( cudaMalloc( ( void**)&d_data_spat , size_ * sizeof( complex_t)));

	CUDA_CHECK( cudaMalloc( ( void**)&d_odata			, size_scaled_ * sizeof( float)));

	CUDA_CHECK( cudaMalloc( ( void**)&d_maps	  , NO_OF_ORIENTS*NO_OF_BANDS * size_ * sizeof( float)));
	CUDA_CHECK( cudaMalloc( ( void**)&d_maps_freq , NO_OF_ORIENTS*NO_OF_BANDS * size_ * sizeof( complex_t)));	
	CUDA_CHECK( cudaMalloc( ( void**)&d_maps_spat , NO_OF_ORIENTS*NO_OF_BANDS * size_ * sizeof( complex_t)));
}


/// <summary>Computes static saliency map.</summary>
/// <param name="out">Destination image.</param>
/// <param name="in">Source image.</param>
/// <param name="im_size">Image size.</param>
/// <returns>Returns the dynamic saliency map.</returns>
/// <remarks>
/// The function computes static saliency map for STVS model. 
/// It takes an input video frames or image, and computes the salience map as:
/// Step 1: Apply the retinal filtering.
/// Step 2: Apply the mask.
/// Step 3: Move to frequency domain.
/// Step 4: Apply the Gabor filtering.
/// Step 5: Move back to spatial domain.
/// Step 6: Perform the short interactions.
/// Step 7: Perform the normalizations.
/// Step 8: Apply the mask.
/// </remarks>
void Static::Pathway::Apply(
							float *out
							, float *in
							, siz_t im_size_scaled
							, siz_t im_size
							)
{	
	assert(  in != NULL);
	assert( out != NULL);

	int INPUT_SIZE, OUTPUT_SIZE;

	INPUT_SIZE	= im_size.w*im_size.h * sizeof( float);
	OUTPUT_SIZE = im_size_scaled.w*im_size_scaled.h * sizeof( float);

	CUDA_CHECK( cudaMemcpy( d_idata, in, INPUT_SIZE, cudaMemcpyHostToDevice));	

		#ifdef VERBOSE
		shrLog("static input ready\n");
#endif

	//oRetina.RetinaFilter( d_data, im_size, 0, 0, 0);

	oMask.Apply( d_data_spat, d_idata, im_size);

		#ifdef VERBOSE
		shrLog("mask\n");
#endif

	oTransform.Apply( d_data_freq, d_data_spat, im_size, CUFFT_FORWARD);

		#ifdef VERBOSE
		shrLog("transform\n");
#endif

	oGabor.Apply( d_maps_freq, d_data_freq, im_size);

		#ifdef VERBOSE
		shrLog("gabor\n");
#endif

	oTransform.Apply( d_maps_spat, d_maps_freq, im_size, CUFFT_INVERSE);

		#ifdef VERBOSE
		shrLog("itransform\n");
#endif

	oInteract.Apply( d_maps, d_maps_spat, im_size);

		#ifdef VERBOSE
		shrLog("interact\n");
#endif

	oNormalize.Apply( d_idata, d_maps, im_size);

		#ifdef VERBOSE
		shrLog("normalize\n");
#endif

	oMask.Apply( d_odata, d_idata, im_size);

		#ifdef VERBOSE
		shrLog("mask\n");
#endif

		#ifdef VERBOSE
		shrLog("resize\n");
#endif

		CUDA_CHECK( cudaDeviceSynchronize());

	CUDA_CHECK( cudaMemcpy( out, d_odata, OUTPUT_SIZE, cudaMemcpyDeviceToHost));

		#ifdef VERBOSE
		shrLog("done\n");
#endif
}


/// <summary>Cleans up the static pathway of STVS model.</summary>
void Static::Pathway::Clean()
{	
	CUDA_CHECK( cudaFree( d_idata));
	CUDA_CHECK( cudaFree( d_odata));
	CUDA_CHECK( cudaFree( d_data_freq));
	CUDA_CHECK( cudaFree( d_data_spat));

	CUDA_CHECK( cudaFree( d_maps));

	CUDA_CHECK( cudaFree( d_maps_freq));	
	CUDA_CHECK( cudaFree( d_maps_spat));

	CUDA_CHECK( cudaThreadExit());

	//_CrtDumpMemoryLeaks();
}