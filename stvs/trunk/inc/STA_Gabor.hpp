
#ifndef _GABOR_H
#define _GABOR_H

#include "../inc/struct.hpp"

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to compute the dynamic visual saliency map for the STVS model.
	/// </summary>
	class Gabor {

	private:

		/// <summary>
		/// Source image size.
		/// </summary>
		siz_t im_size_;

		/// <summary>
		/// No. of pixels in source image.
		/// </summary>
		unsigned int size_;

		void Init();
		void Clean();

		float 
			*h_gaborMaskU
			, *h_gaborMaskV;

		cudaArray 
			*cu_gaborU
			, *cu_gaborV;

		double h_teta[NO_OF_ORIENTS];

		float 
			h_frequencies[NO_OF_BANDS]
		, h_sig_hor[NO_OF_BANDS];

	public:

		/// <summary>
		/// Default contructor for Pathway class.
		/// </summary>
		inline Gabor( siz_t const & im_size)
		{
			im_size_ = im_size;

			size_ = im_size.w*im_size.h;

			Init();
		}

		/// <summary>
		/// Destructor for Pathway class.
		/// </summary>
		inline ~Gabor(){ Clean();}

		void CreateLinear( 
			float *in
			, float d1
			, float d2
			, unsigned int n
			);

		void CreateGaborMasks( 
			float *u1
			, float *v1			
			, siz_t im_size
			, double *teta
			);

		void Apply( 
			complex_t* out
			, complex_t *in
			, siz_t im_size			
			);

	};

	__global__ void KernelGabor( 
		complex_t* out
		, complex_t* in
		, siz_t im_size		
		);

} // namespace Static

#endif // _GABOR_H