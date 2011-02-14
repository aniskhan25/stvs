/*
* Display grayscale video from webcam
*
* Author  anis rahman
*
* 28/01/11 19:58
*/
#include <iostream>

#include "cutil_inline.h"
#include "cv.h"
#include "highgui.h"

#include "error.hpp"
#include "STA_Pathway.hpp"

/**
Main static pathway
*/
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{	
	CUDA_CHECK( cudaSetDevice( 1));

	{
		cudaEvent_t wakeGPU;
		CUDA_CHECK( cudaEventCreate( &wakeGPU));
	}

	float *h_idata, *h_data001, *h_data002, *h_data_mc
		, *h_odata, *h_odata_dyn, *h_odata_sta;

	unsigned int nbPixels;
	siz_t im_size;

	cvNamedWindow( "Color_Image", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "Gray_Image", CV_WINDOW_AUTOSIZE );
	
	CvCapture* capture = cvCaptureFromAVI( "clip1.avi" );
	if (!capture){
		return -1;
	}
	IplImage* bgr_frame;
	double fps = cvGetCaptureProperty (
		capture,
		CV_CAP_PROP_FPS
		);
	printf("fps=%d\n",(int)fps);

	CvSize size = cvSize(
		(int)cvGetCaptureProperty( capture,
		CV_CAP_PROP_FRAME_WIDTH),
		(int)cvGetCaptureProperty( capture,
		CV_CAP_PROP_FRAME_HEIGHT)
		);

	printf("frame (w, h) = (%d, %d)\n", size.width, size.height);

	IplImage* gray_frame = cvCreateImage(
		size,
		IPL_DEPTH_8U,
		1
		);

	im_size.h = size.height;
	im_size.w = size.width;

	nbPixels = im_size.w * im_size.h;

	h_idata	  =( float *)calloc( nbPixels, sizeof( float));
	h_odata	=( float *)calloc( nbPixels, sizeof( float));	

	float *DBG_Float =( float *)calloc( nbPixels, sizeof( float));	
	complex_t *DBG_Complex =( complex_t *)calloc( nbPixels, sizeof( complex_t));	

	Static::Pathway oPathway( im_size);

	while( (bgr_frame=cvQueryFrame(capture)) != NULL ) {
		cvShowImage( "Color_Image", bgr_frame );

		cvCvtColor( bgr_frame, gray_frame, CV_RGB2GRAY );

		for( unsigned int i=0;i<nbPixels;++i){
			h_idata[i] = gray_frame->imageData[i];
		}

		oPathway.Apply(h_idata, im_size, h_odata);

		float max_s;
		max_s = h_odata[0];

		for( unsigned int i=0;i<nbPixels;++i) {
			if( h_odata[i] > max_s)
				max_s = h_odata[i];
		}

		for( unsigned int i=0;i<nbPixels;++i) {
			gray_frame->imageData[i] =( char)( unsigned char)( h_odata[i]/max_s * 255.0f);
		}

		cvShowImage("Gray_Image", gray_frame);

		char c = cvWaitKey(30);
		if( c == 27 ) break;
	}

	cvReleaseImage( &gray_frame );
	cvReleaseCapture( &capture );

	free( h_odata);
	free( h_idata);

	free( DBG_Float);
	free( DBG_Complex);	

	cutilExit( argc, argv);
}
