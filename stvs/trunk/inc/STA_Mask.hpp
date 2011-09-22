#ifndef _MASK_H
#define _MASK_H

// includes
#include "../inc/struct.hpp"

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to compute and apply the hanning mask to discard the borders.
	/// </summary>
	class Mask {

	private:
		/// <summary>
		/// Source image size.
		/// </summary>
		siz_t im_size_;

		/// <summary>
		/// No. of pixels in source image.
		/// </summary>
		unsigned int size_;

		/// <summary>
		/// Hanning mask.
		/// </summary>
		float *h_mask;

		/// <summary>
		/// Defines array variable to bind to gpu's texture memory.
		/// </summary>
		cudaArray *cu_mask;

		void Init();
		void Clean();

	public:

		/// <summary>
		/// Default contructor for Mask class.
		/// </summary>
		inline Mask( siz_t const & im_size)
		{ 
			im_size_ = im_size;
			size_ = im_size_.w*im_size_.h;

			Init();
		}

		/// <summary>
		/// Destructor for Mask class.
		/// </summary>
		inline ~Mask(){ Clean();}

		void CreateMask( 
			float *mask
			, siz_t im_size
			, int mu
			);

		void Apply( 
			float *out
			, float *in
			, siz_t const & im_size			
			);

		void Apply( 
			complex_t *out
			, float *in
			, siz_t const & im_size			
			);

	}; // class Mask		


	__global__ void MaskKernel(
		float *out
		, float* in
		, siz_t im_size		
		);

	__global__ void MaskKernel( 
		complex_t *out
		, float* in
		, siz_t im_size		
		);

} // namespace Static

#endif // _MASK_H