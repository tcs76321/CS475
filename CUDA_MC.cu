// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE		64		// number of threads per block
#endif

//#ifndef SIZE
//#define SIZE			1*1024*1024	// array size
//#endif

#ifndef NUMTRIALS
#define NUMTRIALS		512000		// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif


//helper stuff
float Ranf(float low, float high)
{
	float r = (float)rand();               // 0 - RAND_MAX
	float t = r / (float)RAND_MAX;       // 0. - 1.

	return   low + t * (high - low);
}

int Ranf(int ilow, int ihigh)
{
	float low = (float)ilow;
	float high = ceil((float)ihigh);

	return (int)Ranf(low, high);
}



// ranges for the random numbers:
const float XCMIN = 0.0;
const float XCMAX = 2.0;
const float YCMIN = 0.0;
const float YCMAX = 2.0;
const float RMIN = 0.5;
const float RMAX = 2.0;


// (CUDA Kernel) on the device

__global__  void MonteCarlo( float *xcs, float *ycs, float *rs, float *hits )
{

	//unsigned int numItems = blockDim.x;
	//unsigned int tnum = threadIdx.x;
	//unsigned int wgNum = blockIdx.x;
	//unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
	
	float xc = xcs[gid];
	float yc = ycs[gid];
	float  r = rs[gid];

	// solve for the intersection using the quadratic formula:
	float a = 2.;
	float b = -2.*(xc + yc);
	float c = xc * xc + yc * yc - r * r;
	float d = b * b - 4.*a*c;

	if (d >= 0.0) {
		// hits the circle:
		// get the first intersection:
		d = sqrt(d);
		float t1 = (-b + d) / (2.*a);	// time to intersect the circle
		float t2 = (-b - d) / (2.*a);	// time to intersect the circle
		float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

		if (tmin >= 0.0) {
			//Did not engulf laser

			// where does it intersect the circle?
			float xcir = tmin;
			float ycir = tmin;

			// get the unitized normal vector at the point of intersection:
			float nx = xcir - xc;
			float ny = ycir - yc;
			float n = sqrt(nx*nx + ny * ny);
			nx /= n;	// unit vector
			ny /= n;	// unit vector

			// get the unitized incoming vector:
			float inx = xcir - 0.;
			float iny = ycir - 0.;
			float in = sqrt(inx*inx + iny * iny);
			inx /= in;	// unit vector
			iny /= in;	// unit vector

			// get the outgoing (bounced) vector:
			float dot = inx * nx + iny * ny;
			//float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
			float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

			// find out if it hits the infinite plate:
			float t = (0. - ycir) / outy;

			if (t >= 0.0) {
				//beam went down
				hits[gid] = 1;
			}
		}
	}
	
}


// main program:

int
main( int argc, char* argv[ ] )
{
	int dev = findCudaDevice(argc, (const char **)argv);


	// better to define these here so that the rand() calls don't get into the thread timing:
	float * xcs = new float[NUMTRIALS];
	float * ycs = new float[NUMTRIALS];
	float * rs = new float[NUMTRIALS];
	float * hits = new float[NUMTRIALS];


	// fill the random-value arrays:
	for (int n = 0; n < NUMTRIALS; n++)
	{
		xcs[n] = Ranf(XCMIN, XCMAX);
		ycs[n] = Ranf(YCMIN, YCMAX);
		rs[n] = Ranf(RMIN, RMAX);
		hits[n] = 0;
	}



	// allocate device memory:

	float *d_xcs, *d_ycs, *d_rs, *d_hits;

	dim3 dims_xcs( NUMTRIALS, 1, 1 );
	dim3 dims_ycs( NUMTRIALS, 1, 1 );
	dim3 dims_rs( NUMTRIALS, 1, 1 );
	dim3 dims_hits(NUMTRIALS, 1, 1);

	//__shared__ float prods[SIZE/BLOCKSIZE];


	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&d_xcs), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&d_ycs), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&d_rs), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&d_hits), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );


	// copy host memory to the device:

	status = cudaMemcpy( d_xcs, xcs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( d_ycs, ycs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( d_rs, rs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( d_hits, hits, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( NUMTRIALS / threads.x, 1, 1 );


	// Create and start timer
	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:


	
	MonteCarlo<<< grid, threads >>>( d_xcs, d_ycs, d_rs, d_hits );
	

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double TrialsPerSecond = (float)NUMTRIALS / secondsTotal;
	double megaTrialsPerSecond = TrialsPerSecond / 1000000.;
	fprintf( stderr, "NUMTRIALS = %10d, BLOCKSIZE = %d, MegaTrials/Second = %10.2lf\n", NUMTRIALS, BLOCKSIZE, megaTrialsPerSecond );

	// copy result from the device to the host:

	status = cudaMemcpy( hits, d_hits, ((NUMTRIALS) * sizeof(float)), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the sum :

	float prob = 0.0;

	for (int z = 0; z < NUMTRIALS ;z++) {
		if (hits[z] == 1 ) {
			prob = prob + 1;
		}
	}

	prob = prob / NUMTRIALS;

	float probPerc = prob * 100;

	fprintf( stderr, "Probability as percent: %10.2lf\n\n", probPerc);



	// clean up memory:
	delete [ ] xcs;
	delete [ ] ycs;
	delete [ ] rs;
	delete [ ] hits;

	status = cudaFree( d_xcs );
		checkCudaErrors( status );
	status = cudaFree( d_ycs );
		checkCudaErrors( status );
	status = cudaFree( d_rs );
		checkCudaErrors( status );
	status = cudaFree(d_hits);
		checkCudaErrors(status);


	return 0;
}





/*

// Array multiplication: C = A * B:

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

#ifndef SIZE
#define SIZE			1*1024*1024	// array size
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		100		// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// array multiplication (CUDA Kernel) on the device: C = A * B

__global__  void ArrayMul( float *A, float *B, float *C )
{
	__shared__ float prods[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	prods[tnum] = A[gid] * B[gid];

	for (int offset = 1; offset < numItems; offset *= 2)
	{
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0)
		{
			prods[tnum] += prods[tnum + offset];
		}
	}

	__syncthreads();
	if (tnum == 0)
		C[wgNum] = prods[0];
}


// main program:

int
main( int argc, char* argv[ ] )
{
	int dev = findCudaDevice(argc, (const char **)argv);

	// allocate host memory:

	float * hA = new float [ SIZE ];
	float * hB = new float [ SIZE ];
	float * hC = new float [ SIZE/BLOCKSIZE ];

	for( int i = 0; i < SIZE; i++ )
	{
		hA[i] = hB[i] = (float) sqrt(  (float)(i+1)  );
	}

	// allocate device memory:

	float *dA, *dB, *dC;

	dim3 dimsA( SIZE, 1, 1 );
	dim3 dimsB( SIZE, 1, 1 );
	dim3 dimsC( SIZE/BLOCKSIZE, 1, 1 );

	//__shared__ float prods[SIZE/BLOCKSIZE];


	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dA), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dB), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dC), (SIZE/BLOCKSIZE)*sizeof(float) );
		checkCudaErrors( status );


	// copy host memory to the device:

	status = cudaMemcpy( dA, hA, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dB, hB, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( SIZE / threads.x, 1, 1 );

	// Create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:

	for( int t = 0; t < NUMTRIALS; t++)
	{
	        ArrayMul<<< grid, threads >>>( dA, dB, dC );
	}

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double multsPerSecond = (float)SIZE * (float)NUMTRIALS / secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	fprintf( stderr, "Array Size = %10d, MegaMultReductions/Second = %10.2lf\n", SIZE, megaMultsPerSecond );

	// copy result from the device to the host:

	status = cudaMemcpy( hC, dC, (SIZE/BLOCKSIZE)*sizeof(float), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the sum :

	double sum = 0.;
	for(int i = 0; i < SIZE/BLOCKSIZE; i++ )
	{
		//fprintf(stderr, "hC[%6d] = %10.2f\n", i, hC[i]);
		sum += (double)hC[i];
	}
	fprintf( stderr, "\nsum = %10.2lf\n", sum );

	// clean up memory:
	delete [ ] hA;
	delete [ ] hB;
	delete [ ] hC;

	status = cudaFree( dA );
		checkCudaErrors( status );
	status = cudaFree( dB );
		checkCudaErrors( status );
	status = cudaFree( dC );
		checkCudaErrors( status );


	return 0;
}

*/