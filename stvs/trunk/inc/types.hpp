
#ifndef TYPES_H
#define TYPES_H

//#define DBG_ON	1

#define _PI  3.141592653589793
#define _2PI 6.283185307

#define  EPS		2.2204e-016

#define MIN_EPSILON_ERROR 5e-3f

#define  SIG		12

#define  NO_OF_BANDS		4
#define  NO_OF_ORIENTS		6

#define  STD_MAX	0.25f / 0.125f * 1.0f /( 2.0f * 3.1416f * 3.9f)
#define  FREQ_MAX	0.25f

#define  SCALE		2.0f

/* Here are the values of parameters used by the program */
#define K 3			/* defines the number of levels in a pyramid of frames */
#define N 6			/* defines the number of Gabor-like filters in a bank */
#define VP 15		/* threshold */
#define f0 0.125	// frequency for gabor-like filter
#define s0 3.9		// scale for gabor-like filter
#define s1 2.2		// scale for high-pass filter 
#define s2 3.5		// scale for motion matrices filter
#define s3 1.0		// scale for pyramid filter

#define Pmax 3		// maximum number of iteration for solving the oversized system
#define Dmin 0.01	// minimum difference between two iterations needed to break the loop
#define C 0.5		// parameter of biweight estimator
#define Mmax 3		// maximum value estimated which is not avoided( for any pyramid level)
					//( put zero or a negative value if you don't want to use it, 
					//  in that case you may change the value of C)

#define KERNEL_PHOTO 15 // kernel size for photoreceptor cells

#endif /* TYPES_H */