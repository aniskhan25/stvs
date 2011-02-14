
#ifndef _TRANSFORM_H
#define _TRANSFORM_H

#include "common.hpp"

namespace Static {

	class Transform {

	private:

		siz_t _im_size;

		unsigned int size;

		void Init();
		void Clean();

	public:

		inline Transform( siz_t const & im_size){ _im_size = im_size; size = im_size.w*im_size.h; Init(); }
		inline ~Transform(){ Clean();}

		void FFT( complex_t*, siz_t, complex_t*, int);
		
		void Apply( complex_t* in, siz_t _im_size, complex_t* out, int direction);

	};

	__global__ void FFTShiftKernel( complex_t* in, siz_t im_size, unsigned int centerW, unsigned int centerH, unsigned int widthEO, unsigned int heightEO, complex_t* out);
	__global__ void IFFTShiftKernel( complex_t* inMaps, siz_t im_size, unsigned int centerW, unsigned int centerH, unsigned int widthEO, unsigned int heightEO, complex_t* outMaps);

} // namespace Static

#endif // _TRANSFORM_H