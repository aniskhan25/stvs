
#ifndef _PATHWAY_H
#define _PATHWAY_H

#include "STA_Mask.hpp"
#include "STA_Transform.hpp"
#include "STA_Gabor.hpp"
#include "STA_Interact.hpp"
#include "STA_Normalize.hpp"

namespace Static {
	
	class Pathway {

	private:

		siz_t _im_size;

		unsigned int size;	

		void Init();
		void Clean();

	public:
		// Input
		float *d_data, *d_mapsout;

		complex_t *d_dataComplex, *d_dataFFTShift, *d_maps, *d_mapsShifted;

		inline Pathway( siz_t const & im_size): oMask( im_size), oTransform( im_size), oGabor( im_size), oInteract( im_size), oNormalize( im_size) {_im_size.w = im_size.w; _im_size.h = im_size.h; size = _im_size.w*_im_size.h; Init(); }
		inline ~Pathway(){ Clean(); }

		Mask oMask;
		Transform oTransform;
		Gabor oGabor;
		Interact oInteract;
		Normalize oNormalize;		
				
		void Apply( float*, siz_t, float*);
		
		void SetData( float* h_data, float* d_data);

		void GetData( float* d_data, float* h_data);
		void GetData( complex_t* d_data, complex_t* h_data);
	};

} // namespace Static

#endif