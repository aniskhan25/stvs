
#ifndef _INTERACT_H
#define _INTERACT_H

#include "../inc/struct.hpp"

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to compute the interactions among the feature maps.
	/// </summary>
	class Interact {

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

	public:

		/// <summary>
		/// Default contructor for Interact class.
		/// </summary>
		inline Interact( siz_t const & im_size)
		{
			im_size_ = im_size;
			size_ = im_size.w*im_size.h;

			Init();
		}

		/// <summary>
		/// Destructor for Interact class.
		/// </summary>
		inline ~Interact(){ Clean();}

		void Apply(
			float* out
			, complex_t* in
			, siz_t im_size			
			);

	};

	__global__ void KernelInteractionShort( 
		float* out
		, complex_t* in
		, siz_t im_size		
		);

} // namespace Static

#endif // _INTERACT_H