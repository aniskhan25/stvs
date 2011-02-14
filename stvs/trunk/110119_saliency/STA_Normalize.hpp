
#ifndef _NORMALIZE_H
#define _NORMALIZE_H

#include "common.hpp"

#include "STA_Reduce.hpp"

namespace Static {

	class Normalize {

		siz_t _im_size;

		unsigned int size;

		float *d_oSum, *d_oMax, *d_oMin;

		void Init();
		void Clean();

	public:
		
		inline Normalize( siz_t const & im_size){ _im_size = im_size; size = im_size.w*im_size.h; Init(); }
		inline ~Normalize(){ Clean(); }
				
		void Apply( float*, siz_t, float*);

		Static::Reduce oReduce;
		
	};

	__global__ void NormNLKernel( float* mapsout, float* maxpt, float* minpt, siz_t im_size, int blocks);

	__global__ void NormIttiKernel( float* mapsout, float* sumpt, float* maxpt, siz_t im_size, int blocks);

	__global__ void NormPCFusionKernel( float* mapsout, float* maxpt, siz_t im_size, float* imageout, int blocks);

} // namespace Static

#endif // _NORMALIZE_H