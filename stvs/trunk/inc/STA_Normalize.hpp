
#ifndef _NORMALIZE_H
#define _NORMALIZE_H

// includes
#include "struct.hpp"
#include "STA_Reduce.hpp"

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to apply different normalizations to the feature maps.
	/// </summary>
	class Normalize {

		/// <summary>
		/// Source image size.
		/// </summary>
		siz_t im_size_;

		/// <summary>
		/// No. of pixels in source image.
		/// </summary>
		unsigned int size_;

		/// <summary>
		/// Reduction operator results (host).
		/// </summary>
		summary_stats_data *result;

		/// <summary>
		/// Reduction operator results (device).
		/// </summary>
		summary_stats_data *d_result;

		/// <summary>
		/// Object of Reduce class used for reduction operations.		
		/// </summary>		
		Static::Reduce oReduce;

		void Init();
		void Clean();

	public:

		/// <summary>
		/// Default contructor for Normalize class.
		/// </summary>
		inline Normalize( siz_t const & im_size)
		{
			im_size_ = im_size;
			size_ = im_size.w*im_size.h;

			Init();
		}

		/// <summary>
		/// Destructor for Normalize class.
		/// </summary>
		inline ~Normalize(){ Clean();}

		void Apply( 
			float *out
			, float* in
			, siz_t im_size			
			);

	}; // class normalize

	__global__  void KernelNormalizeNL( 
		float* maps
		, summary_stats_data* result	
		, siz_t im_size	
		);

	__global__  void KernelNormalizeItti(
		float* maps
		, summary_stats_data* result
		, siz_t im_size	
		);

	__global__  void KernelNormalizePCFusion(
		float* out
		, float* maps
		, summary_stats_data* result
		, siz_t im_size	
		);

} // namespace Static

#endif // _NORMALIZE_H