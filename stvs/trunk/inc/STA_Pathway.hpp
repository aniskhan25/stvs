
#ifndef _STA_PATHWAY_H
#define _STA_PATHWAY_H

// includes
#include "../inc/struct.hpp"
#include "../inc/error.hpp"

#include "../inc/STA_Mask.hpp"
#include "../inc/STA_Transform.hpp"
#include "../inc/STA_Gabor.hpp"
#include "../inc/STA_Interact.hpp"
#include "../inc/STA_Normalize.hpp"

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to compute the static visual saliency map for the STVS model.
	/// </summary>
	class Pathway {

	private:

		/// <summary>
		/// Source image size.
		/// </summary>
		siz_t im_size_;

		/// <summary>
		/// Output image size.
		/// </summary>
		siz_t im_size_scaled_;

		/// <summary>
		/// No. of pixels in destination image.
		/// </summary>
		unsigned int size_;

		/// <summary>
		/// No. of pixels in destination image.
		/// </summary>
		unsigned int size_scaled_;

		void Init();
		void Clean();

	public:

		// declarations
		float 
			*d_idata
			, *d_odata
			, *d_maps;

		complex_t 
			*d_data_freq
			, *d_data_spat			
			, *d_maps_freq
			, *d_maps_spat;

		/// <summary>
		/// Default contructor for Pathway class.
		/// </summary>
		inline Pathway( siz_t const & im_size, float const & scale = 1.0f)
			: oMask( im_size)
			, oTransform( im_size)
			, oGabor( im_size)
			, oInteract( im_size)
			, oNormalize( im_size)
		{
			im_size_.w = im_size.w;
			im_size_.h = im_size.h;

			size_ = im_size_.w*im_size_.h;

			im_size_scaled_.w = (int)( im_size.w * scale);
			im_size_scaled_.h = (int)( im_size.h * scale);

			size_scaled_ = im_size_scaled_.w*im_size_scaled_.h;

			Init();
		}

		/// <summary>
		/// Destructor for Pathway class.
		/// </summary>
		inline ~Pathway(){ Clean();}

		/// <summary>Object of Reduce class used for retinal filtering operations.</summary>
		//Retina oRetina;

		/// <summary>Object of Reduce class used for masking operations.</summary>
		Mask oMask;

		/// <summary>Object of Reduce class used for fourier transformations.</summary>
		Transform oTransform;

		/// <summary>Object of Reduce class used for gabor filtering operations.</summary>
		Gabor oGabor;

		/// <summary>Object of Reduce class used for interaction operations.</summary>
		Interact oInteract;

		/// <summary>Object of Reduce class used for normalizations.</summary>
		Normalize oNormalize;		

		void Apply(
			float *out
			, float *in
			, siz_t im_size_scaled
			, siz_t im_size
			);

		void SetData(
			float* out
			, float* in
			);

		void GetData( 
			float* out
			, float* in
			); 

		void GetData( 
			complex_t* out
			, complex_t* in
			);

	}; // class Pathway

} // namespace Static

#endif // _STA_PATHWAY_H