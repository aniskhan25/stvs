
#ifndef _REDUCE_H
#define _REDUCE_H

// includes
#include <limits>
#include "struct.hpp"

// structure used to accumulate the moments and other 
// statistical properties encountered so far.
struct summary_stats_data
{    
	float sum;
	float min;
	float max;

	// initialize to the identity element
	void initialize()
	{
		sum = 0;
		min = std::numeric_limits<float>::max();
		max = std::numeric_limits<float>::min();
	}
};

/// <summary>
/// This namespace wraps the static pathway's functionality.
/// </summary>
namespace Static {

	/// <summary>
	/// A class with functions to compute the reduction operations.
	/// </summary>
	class Reduce {

	private:

	public:

		/// <summary>
		/// Default contructor for Reduce class.
		/// </summary>
		inline Reduce(){}

		/// <summary>
		/// Destructor for Reduce class.
		/// </summary>
		inline ~Reduce(){}

		summary_stats_data Apply( float *data, siz_t im_size );
	}; // class Normalize

} // namespace Static

#endif // _REDUCE_H