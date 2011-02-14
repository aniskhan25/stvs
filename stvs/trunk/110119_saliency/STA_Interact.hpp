
#ifndef _INTERACT_H
#define _INTERACT_H

#include "common.hpp"

namespace Static {

	class Interact {

	private:

		siz_t _im_size;

		unsigned int size;

		 void Init();
		 void Clean();

	public:

		inline Interact(){}
		inline Interact( siz_t const & im_size){ _im_size = im_size; size = im_size.w*im_size.h; Init(); }
		inline ~Interact(){ Clean(); }

		void Apply( complex_t*, siz_t, float*);
	};

	__global__ void ShortInteractionKernel( complex_t* in, siz_t im_size, float* mapsout);

} // namespace Static

#endif // _INTERACT_H