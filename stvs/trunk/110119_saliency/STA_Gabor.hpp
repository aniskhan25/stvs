
#ifndef _GABOR_H
#define _GABOR_H

#include "common.hpp"

namespace Static {

	class Gabor {

	private:

		siz_t _im_size;

		unsigned int size;

		void Init();
		void Clean();

		float *h_gaborMaskU, *h_gaborMaskV;

		cudaArray *cu_gaborU, *cu_gaborV;

		double h_teta[NO_OF_ORIENTS];

		float h_frequencies[NO_OF_BANDS], h_sig_hor[NO_OF_BANDS];

	public:
		
		inline Gabor( siz_t const & im_size){ _im_size = im_size; size = im_size.w*im_size.h; Init(); }
		inline ~Gabor(){ Clean(); }

		void CreateLinear( float *in, float d1, float d2, unsigned int n);

		void CreateGaborMasks( float*, float*, double*, siz_t);

		void Apply( complex_t *in, siz_t im_size, complex_t* out);
	};

	__global__ void GaborKernel( complex_t* in, siz_t im_size, complex_t* maps);

} // namespace Static

#endif // _GABOR_H