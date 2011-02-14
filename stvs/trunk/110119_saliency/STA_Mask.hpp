#ifndef _MASK_H
#define _MASK_H

#include "common.hpp"

namespace Static {

	class Mask {

	private:

		void Init();
		void Clean();

		cudaArray *cu_mask;

		siz_t _im_size;

		unsigned int size;

	public:

		float *h_mask;

		inline Mask( siz_t const & im_size){ _im_size = im_size; size = im_size.w*im_size.h; Init();}
		inline ~Mask(){ Clean(); }

		int CreateMask( float *mask, unsigned int rows, unsigned int cols, int mu);

		void Apply( float*, siz_t const &, float*);
		void Apply( float*, siz_t const &, complex_t*);
	};				


	__global__ void MaskKernel( float* in, siz_t im_size, complex_t *out);
	__global__ void MaskKernel( float* in, siz_t im_size, float *out);

} // namespace Static

#endif // _MASK_H