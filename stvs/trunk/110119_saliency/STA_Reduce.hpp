#ifndef _REDUCE_H
#define _REDUCE_H

namespace Static {

	class Reduce {

	public:

		inline Reduce(){}
		inline ~Reduce(){}

		void reduce6_sum( int size, int threads, int blocks, float *d_idata, float *d_odata);

		void reduce6_max( int size, int threads, int blocks, float *d_idata, float *d_odata);

		void reduce6_min( int size, int threads, int blocks, float *d_idata, float *d_odata);

		void getNumBlocksAndThreads( int n, int maxThreads, int &blocks, int &threads);
	};

	__global__ void reduce6_sum_kernel( float *g_idata, float *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2);
	__global__ void reduce6_max_kernel( float *g_idata, float *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2);
	__global__ void reduce6_min_kernel( float *g_idata, float *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2);

} // namespace Static

#endif // _REDUCE_H