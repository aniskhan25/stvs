
#ifndef _TRANSFORM_H
#define _TRANSFORM_H

// includes
#include "../inc/struct.hpp"

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to apply the fourier transformations.
	/// </summary>
	class Transform {

	private:

		cufftHandle plan;

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

	public:

		/// <summary>
		/// Default contructor for Transform class.
		/// </summary>
		inline Transform( siz_t const & im_size)
		{
			im_size_ = im_size;
			size_ = im_size.w*im_size.h;

			Init();
		}

		/// <summary>
		/// Destructor for Transform class.
		/// </summary>
		inline ~Transform() { Clean(); }

		void FFT( 
			complex_t* out
			, complex_t* in
			, siz_t im_size			
			, int direction
			);

		void Apply( 
			complex_t* out
			, complex_t* in
			, siz_t im_size
			, int direction
			);

	}; // class transform

	__global__ void KernelShift(
		complex_t* out
		, complex_t* in
		, siz_t im_size
		, point_t center
		, bool is_width_odd
		, bool is_height_odd		
		);


	__global__ void KernelShiftInverse(
		complex_t* out
		, complex_t* in
		, siz_t im_size
		, point_t center
		, bool is_width_odd
		, bool is_height_odd		
		);

} // namespace Static

#endif // _TRANSFORM_H