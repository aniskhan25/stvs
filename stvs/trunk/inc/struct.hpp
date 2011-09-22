
#ifndef STRUCT_H
#define STRUCT_H

// includes
#include "../inc/types.hpp"

#ifndef FALSE
#define FALSE false
#endif
#ifndef TRUE
#define TRUE true
#endif

/// <summary>Image size.</summary>
typedef struct siz_t{	

	siz_t(){}
	siz_t(int w_, int h_){ w = w_; h = h_;}

	/// <summary>Width.</summary>
	int w;
	/// <summary>Height.</summary>
	int h;
} siz_t;

/// <summary>Image location.</summary>
typedef struct point_t{	

		point_t(){}
	point_t(int x_, int y_){ x = x_; y = y_;}

	/// <summary>x coordinate.</summary>
	unsigned int x;
	/// <summary>y coordinate.</summary>
	unsigned int y;
} point_t;

typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct{
	float x;
	float y;
} fComplex;
#endif

/// <summary>Complex datatype definition.</summary>
typedef float complex_t[2];

/// <summary>Structure holding the image pyramid level's information.</summary>
typedef struct{

	/// <summary>[matrix] Current frame.</summary>
	float *frame;

	/// <summary>[matrix] Previous frame.</summary>
	float *prev;

	/// <summary>[matrix] Shifted frame.</summary>
	float *shift;

	/// <summary>[bank] Filtered frame.</summary>
	float *filt;

	/// <summary>[bank] Previous frame filtered.</summary>
	float *prefilt;

	/// <summary>[matrix] Motion estimation in x direction.</summary>
	float *vx;

	/// <summary>[matrix] Motion estimation in y direction.</summary>
	float *vy;

	/// <summary>[matrix] Smoothed motion estimation in x direction.</summary>
	float *vx_med;

	/// <summary>[matrix] Smooth motion estimation in y direction.</summary>
	float *vy_med;

	/// <summary>Pyramid level size.</summary>
	siz_t level_size;

} level_t; // end structure pyramid level


/// <summary>Structure holding the image pyramid's information.</summary>
typedef struct {

	/// <summary>[matrix]Pyramid levels.</summary>
	level_t levels[K];	/*  current frame */

	/// <summary>[banks] Spatial and temporal gradients.</summary>
	float *gx, *gy, *gt;

	/// <summary>[matrix] Threshold to eliminate unwanted motion vectors.</summary>
	float *threshold;

	/// <summary>[matrix] Smoothed threshold to eliminate unwanted motion vectors.</summary>
	float *threshold_med;

	/// <summary>TODO: For motion estimator.</summary>
	float *dx, *dy, *wi;

	/// <summary>[bank] modulation matrices.</summary>
	float *mod;
	
	/// <summary>[matrix] Temporary buffer.</summary>
	float *temp;
	
	/// <summary>[matrix] Source image.</summary>
	float *im_data;

	/// <summary>Source image size.</summary>
	siz_t im_size;

} pyramid; // end structure pyramid


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b){
	return (a % b != 0) ?  (a - a % b + b) : a;
}

#define IMUL(a, b) __mul24(a, b)

/*

matrix = new float[w*h]
bank = new complex[w*h] * N

*/

#endif
