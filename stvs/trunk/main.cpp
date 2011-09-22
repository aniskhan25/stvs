
#include "cutil_inline.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "inc/error.hpp"
#include "inc/STA_Pathway.hpp"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{	
	CUDA_CHECK( cudaSetDevice(0));

	{
		cudaEvent_t wakeGPU;
		CUDA_CHECK( cudaEventCreate( &wakeGPU));
	}

	float *h_idata, *h_data001, *h_data002, *h_data_mc
		, *h_odata, *h_odata_dyn, *h_odata_sta;

	unsigned int nbPixels;
	siz_t im_size;

	cv::Mat frame, gray;
	cv::VideoCapture capture; // open the default camera

	cv::namedWindow( "Input", CV_WINDOW_AUTOSIZE );
	cv::namedWindow( "Static", CV_WINDOW_AUTOSIZE );

	capture.open("ClipRMX_3.avi"); // open the default camera

	if(!capture.isOpened()) { // check if we succeeded
		printf( "Unable to read input video." );
		return -1;
	}

	capture >> frame; // get a new frame from camera

	cv::cvtColor(frame, gray, CV_RGB2GRAY);

	im_size.h = gray.rows;
	im_size.w = gray.cols;

	nbPixels = im_size.w * im_size.h;

	h_idata	= ( float *)calloc( nbPixels, sizeof( float));
	h_odata	= ( float *)calloc( nbPixels, sizeof( float));	

	Static::Pathway oStaticPathway( im_size);

	while(1){

		for( unsigned int i=0;i<nbPixels;++i){
			h_idata[i] = gray.data[i];
		}

		oStaticPathway.Apply( h_odata, h_idata, im_size, im_size);

		float mx = h_odata[0];
		for( int i=0;i<nbPixels;++i){
			if (h_odata[i]>mx){
				mx = h_odata[i];
			}
		}

		for( int i=0;i<nbPixels;i++){
			gray.data[i] = ( char)( unsigned char)( h_odata[i]/mx*255.0f);
		}

		cv::imshow("Static", gray);
		char c = cv::waitKey(30);

		if( c == 27 ) break;

		// READ new frame from camera
		capture >> frame;
		cv::cvtColor(frame,gray,CV_RGB2GRAY);

		cv::imshow("Input", gray);
		cv::waitKey(30);
	}

	free( h_odata);
	free( h_idata);

	cutilExit( argc, argv);
}