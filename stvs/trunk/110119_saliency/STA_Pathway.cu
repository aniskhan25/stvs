#include "error.hpp"
#include "cufft_common.hpp"

#include "STA_Pathway.hpp"

void Static::Pathway::SetData( float* h_data, float* d_data) {

	CUDA_CHECK( cudaMemcpy( d_data, h_data, size * sizeof( float), cudaMemcpyHostToDevice));
}

void Static::Pathway::GetData( float* d_data, float* h_data) {

	CUDA_CHECK( cudaMemcpy( h_data, d_data, size*sizeof( float), cudaMemcpyDeviceToHost));
}

void Static::Pathway::GetData( complex_t* d_data, complex_t* h_data) {

	CUDA_CHECK( cudaMemcpy( h_data, d_data, size*sizeof( complex_t), cudaMemcpyDeviceToHost));
}

void Static::Pathway::Apply( float *in, siz_t im_size, float *out)
{
	dim3 dimBlock, dimGrid;

	unsigned int nbPixels = _im_size.w*_im_size.h;

	/**
	Copy input to device memory
	*/	
	CUDA_CHECK( cudaMemcpy( d_data, in, size * sizeof( float), cudaMemcpyHostToDevice));	

	/**
	Retina filter
	*/
	//DYN_retina_filter( d_data, im_size.h, im_size.w, 0, 0, 0);

	/**
	Pre mask
	*/	
	oMask.Apply( d_data, _im_size, d_dataComplex);

	/**
	Shift to frequency domain
	*/
	oTransform.Apply( d_dataComplex, _im_size, d_dataFFTShift, CUFFT_FORWARD);

	/**
	Gabor filter bank
	*/
	oGabor.Apply( d_dataFFTShift, _im_size, d_maps);

	/**
	Shift back to spatial domain
	*/
	oTransform.Apply( d_maps, _im_size, d_mapsShifted, CUFFT_INVERSE);

	/**
	Interaction
	*/
	oInteract.Apply( d_mapsShifted, _im_size, d_mapsout);

	/**
	Normalizations, and finally fusion
	*/
	oNormalize.Apply( d_mapsout, _im_size, d_data);

	/**
	Post mask
	*/
	oMask.Apply( d_data, _im_size, d_data);

	/**
	Copy back output from device memory
	*/
	CUDA_CHECK( cudaMemcpy( out, d_data, size*sizeof( float), cudaMemcpyDeviceToHost));
}

void Static::Pathway::Init()
{
	// Input	
	CUDA_CHECK( cudaMalloc(( void**)&d_data, size * sizeof( float)));

	// complex_t Input	
	CUDA_CHECK( cudaMalloc(( void**)&d_dataComplex, size * sizeof( complex_t)));

	// Gabor
	CUDA_CHECK( cudaMalloc(( void**)&d_maps, NO_OF_ORIENTS*NO_OF_BANDS * size * sizeof( complex_t)));

	CUDA_CHECK( cudaMalloc(( void**)&d_mapsout, NO_OF_ORIENTS*NO_OF_BANDS * size * sizeof( float)));

	// fftShift
	CUDA_CHECK( cudaMalloc(( void**)&d_dataFFTShift, size * sizeof( complex_t)));

	// Inverse Shift
	CUDA_CHECK( cudaMalloc(( void**)&d_mapsShifted, NO_OF_ORIENTS*NO_OF_BANDS * size * sizeof( complex_t)));
}

/**
Cleanup
*/
void Static::Pathway::Clean()
{	
	CUDA_CHECK( cudaFree( d_data));
	CUDA_CHECK( cudaFree( d_dataComplex));
	CUDA_CHECK( cudaFree( d_maps));
	CUDA_CHECK( cudaFree( d_mapsout));
	CUDA_CHECK( cudaFree( d_dataFFTShift));
	CUDA_CHECK( cudaFree( d_mapsShifted));

	CUDA_CHECK( cudaThreadExit());

	//_CrtDumpMemoryLeaks();
}