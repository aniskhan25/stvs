#ifndef _GPUDEFINES_H_
#define _GPUDEFINES_H_

#ifdef COMPILE_FOR_GPU
#       include <cuda.h>
#       ifndef GPU_DEVICE
#               define GPU_DEVICE __device__
#       endif
#       ifndef GPU_HOST
#               define GPU_HOST __host__
#       endif
#       ifndef GPU_HOST_AND_DEVICE
#               define GPU_HOST_AND_DEVICE __device__ __host__
#       endif
#else
#       ifndef GPU_DEVICE
#               define GPU_DEVICE
#       endif
#       ifndef GPU_HOST
#               define GPU_HOST
#       endif
#       ifndef GPU_HOST_AND_DEVICE
#               define GPU_HOST_AND_DEVICE
#       endif
#endif

#endif