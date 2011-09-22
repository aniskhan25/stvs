
// includes
#include "../inc/error.hpp"
#include "../inc/STA_Interact.hpp"


/// <summary>GPU kernel to apply the short interactions.</summary>
/// <param name="out">Interacted Gabor filtered maps.</param>
/// <param name="in">Source image Gabor filtered maps.</param>
/// <param name="im_size">Source image size.</param>
/// <returns>Returns interacted Gabor filtered maps</returns>
/// <remarks>
/// Stage 1: Convert and prefetch complex to real numbers.
/// Stage 2: Perform interactions.
/// </remarks>
__global__  void Static::KernelInteractionShort( 
	float* out
	,complex_t* in
	, siz_t im_size	
	) 
{
	unsigned int width  = im_size.w;
	unsigned int height = im_size.h;

	unsigned int x   = blockIdx.x*blockDim.x + threadIdx.x/2;
	unsigned int x2  = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y   = blockIdx.y*blockDim.y + threadIdx.y;

	if( (x+16)>=width || x2>=width || y>=height) return;

	unsigned int mod = threadIdx.x % 2;

	__shared__ float maps[NO_OF_ORIENTS*NO_OF_BANDS][32];
	__shared__ float buf [72];

	/// <summary>
	/// A pitch of 8 is used to avoid shared memory bank conflicts.
	/// Each shared memory location has a bank reserved for its access. 
	/// In the case, when two memory location are allocated the same bank, and this bank is asked in parallel by the respective threads to access the memory location — a bank conflict occurs.
	/// </summary>
	unsigned int pt = threadIdx.x/2	+ 40*mod;

	/// <summary>
	/// In the 1st stage, the code converts complex to real products, and stores them in shared memory as prefetched data for the second stage.
	/// For coalesced memory accesses the threads process complex numbers in parallel and store them with real and imaginary portions interlaced.
	/// Step 1: 32 threads process 16 complex numbers.
	/// ***************************************************************************************************
	///				16 real no.					16 real no.									-	
	/// |0------------------------------|16---------------------------31|					 |
	/// |R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R| | | | | | | | | | | | | | | | |					 |	
	/// |32-------------|40-----------------------------|56---------------------------71|	  >	buf[72]
	/// |-|-|-|-|-|-|-|-|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C| | | | | | | | | | | | | | | | |	 |
	/// |---------------|-------------------------------|-------------------------------|	 |
	///		pitch = 8			16 imaginary no.				16 imaginary no.			-
	/// ***************************************************************************************************
	/// Step 2: 32 threads process next 16 complex numbers.
	/// ***************************************************************************************************
	///				16 real no.					16 real no.									-	
	/// |0------------------------------|16---------------------------31|					 |
	/// |R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|					 |	
	/// |32-------------|40-----------------------------|56---------------------------71|	  >	buf[72]
	/// |-|-|-|-|-|-|-|-|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|C|	 |
	/// |---------------|-------------------------------|-------------------------------|	 |
	///		pitch = 8			16 imaginary no.				16 imaginary no.			-
	/// ***************************************************************************************************
	/// Step 3: Synchronize the shared memory with interlaced 32 complex numbers.
	/// Step 4: 32 threads produce 32 real products.
	/// ***************************************************************************************************
	///						32 real products												-	
	/// |0------------------------------------------------------------31|					 |
	/// |R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|					 |	
	/// |R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|					  >	maps[*][32]
	/// |							  ...								|					 |
	/// |R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|R|					 |
	/// |---------------------------------------------------------------|					-	
	/// ***************************************************************************************************
	/// </summary>
	for( unsigned int j = 0;j < NO_OF_ORIENTS;++j) {
		for( unsigned int i = 0;i < NO_OF_BANDS;++i) {

			buf[pt	 ] = in[(j*NO_OF_BANDS + i) * ( width*height) +( y*width + x	 )][mod];			
			buf[pt+16] = in[(j*NO_OF_BANDS + i) * ( width*height) +( y*width + x + 16)][mod];

			__syncthreads();
						
			maps[j*NO_OF_BANDS + i][threadIdx.x] = 			
				abs( buf[threadIdx.x]*buf[threadIdx.x] + buf[40 + threadIdx.x]*buf[40 + threadIdx.x]);
		} 
	}

	__syncthreads();

	unsigned int jp, jm;

	/// <summary>
	/// In the 2nd stage, the prefetched data interacts with the other image maps.
	/// Such that values in same orientation causes excitation, whereas ones in the same frequency band causes inhibition.
	/// </summary>
	for( unsigned int j = 0;j < NO_OF_ORIENTS;j++) {
		for( unsigned int i = 0;i < NO_OF_BANDS;i++) {

			jp = j + 1;
			jm = j - 1;

			if( j == NO_OF_ORIENTS-1) jp = 0;
			if( j == 0				) jm = NO_OF_ORIENTS-1;

			if( i == 0			) {
				out[( j*NO_OF_BANDS + i) *( width*height) +( y*width + x2)] =
					maps[ j*NO_OF_BANDS + i    ][threadIdx.x] +				
					0.50f * maps[ j*NO_OF_BANDS + i + 1][threadIdx.x] -
					0.25f * maps[jp*NO_OF_BANDS + i    ][threadIdx.x] -
					0.25f * maps[jm*NO_OF_BANDS + i    ][threadIdx.x];
			}
			else if( i == NO_OF_BANDS-1) {
				out[( j*NO_OF_BANDS + i) *( width*height) +( y*width + x2)] =
					maps[ j*NO_OF_BANDS + i    ][threadIdx.x] +
					0.50f * maps[ j*NO_OF_BANDS + i - 1][threadIdx.x] -
					0.25f * maps[jp*NO_OF_BANDS + i    ][threadIdx.x] -
					0.25f * maps[jm*NO_OF_BANDS + i    ][threadIdx.x];
			}	
			else {
				out[( j*NO_OF_BANDS + i) *( width*height) +( y*width + x2)] =
					maps[ j*NO_OF_BANDS + i    ][threadIdx.x] +
					0.50f * maps[ j*NO_OF_BANDS + i - 1][threadIdx.x] +
					0.50f * maps[ j*NO_OF_BANDS + i + 1][threadIdx.x] -
					0.50f * maps[jp*NO_OF_BANDS + i    ][threadIdx.x] -
					0.50f * maps[jm*NO_OF_BANDS + i    ][threadIdx.x];
			}
		}	
	}  
}


/// <summary>Applies short interactions.</summary>
/// <param name="out">Interacted Gabor filtered maps.</param>
/// <param name="in">Source image Gabor filtered maps.</param>
/// <param name="im_size">Source image size.</param>
/// <returns>Returns interacted Gabor filtered maps</returns>
void Static::Interact::Apply(
							 float* out
							 , complex_t* in
							 , siz_t im_size							 
							 ) 
{
	dim3 dimBlock( 32, 1);	
	dim3 dimGrid( 
		iDivUp( im_size.w, dimBlock.x)
		, iDivUp( im_size.h, dimBlock.y)		
		);

	Static::KernelInteractionShort<<< dimGrid, dimBlock, 0 >>>( out, in, im_size);
	CUDA_CHECK( cudaDeviceSynchronize());
}


/// <summary>Initializes the interactions.</summary>
void Static::Interact::Init(){}


/// <summary>Cleans up the interactions.</summary>
void Static::Interact::Clean(){}