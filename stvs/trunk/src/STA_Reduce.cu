#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <cmath>

#include "../inc/STA_Reduce.hpp"

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
struct summary_stats_unary_op
{
	__host__ __device__
		summary_stats_data operator()(const float& x) const
	{
		summary_stats_data result;
		result.sum	 = x;
		result.min  = x;
		result.max  = x;
		
		return result;
	}
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data 
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for 
// all values that have been agregated so far
struct summary_stats_binary_op 
	: public thrust::binary_function<const summary_stats_data&, 
	const summary_stats_data&,
	summary_stats_data>
{
	__host__ __device__
		summary_stats_data operator()(const summary_stats_data& x, const summary_stats_data & y) const
	{
		summary_stats_data result;

		//Basic number of samples (n), min, and max		
		result.sum = x.sum + y.sum;
		result.min = thrust::min(x.min, y.min);
		result.max = thrust::max(x.max, y.max);

		return result;
	}
};

summary_stats_data Static::Reduce::Apply( float *data, siz_t im_size )
{
	// transfer to device
	thrust::device_ptr<float> d_x(data);

	// setup arguments
	summary_stats_unary_op unary_op;
	summary_stats_binary_op binary_op;
	summary_stats_data init;

	init.initialize();

	// compute summary statistics
	return thrust::transform_reduce(d_x, d_x + (im_size.w*im_size.h), unary_op, init, binary_op);
}