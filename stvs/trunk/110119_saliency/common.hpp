
#ifndef TYPES_H
#define TYPES_H

#define PI				3.141592653589793

#define NO_OF_BANDS		4
#define NO_OF_ORIENTS	6

#define STD_MAX			0.25f / 0.125f * 1.0f /( 2.0f * 3.1416f * 3.9f)
#define FREQ_MAX		0.25f

#define SCALE			2.0f

#define MAX_THREADS		128

#ifndef FALSE
#  define FALSE		0
#endif
#ifndef TRUE
#  define TRUE		1
#endif

typedef struct{	
	unsigned int w;
	unsigned int h;
} siz_t;

typedef float complex_t[2];

#endif /* TYPES_H */