#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h> // for timing

#define ARRAYSIZE 1000

#define NUMTRIES 10

#define SSE_WIDTH 4

void SimdMul( float *a, float *b,   float *c,   int len );

float SimdMulSum( float *a, float *b, int len );

float NonSimdMulSum( float *a, float *b, int len );

void NonSimdMul(float *a, float *b,   float *c,   int len);

//Create a, b, c arrays
float a [ARRAYSIZE];
float b [ARRAYSIZE];
float c [ARRAYSIZE];

int main(){
    
    //Fill a and b
    for(int i = 0; i < ARRAYSIZE ;i++){
        a[i] = 4;
        b[i] = 6;
    }
    
    //
    int len = ARRAYSIZE;
    float sum1 = 0.0;
    float sum2 = 0.0;
    
    double maxMegaMults_SimdMul = 0.0;
    double maxMegaMults_NonSimdMul = 0.0;
    double maxMegaMults_SimdMulSum = 0.0;
    double maxMegaMults_NonSimdMulSum = 0.0;
    
    int t;
    
    for(t = 0; t < NUMTRIES; t++){
        double time0 = omp_get_wtime();
        SimdMul(a,b,c,len);
        double time1 = omp_get_wtime();
        double megaMults = (double)ARRAYSIZE/(time1-time0)/1000000.;
        if(megaMults > maxMegaMults_SimdMul){
            maxMegaMults_SimdMul = megaMults;
        }
    }
    
    
    for(t = 0; t < NUMTRIES; t++){
        double time2 = omp_get_wtime();
        NonSimdMul(a,b,c,len);
        double time3 = omp_get_wtime();
        double megaMults = (double)ARRAYSIZE/(time3-time2)/1000000.;
        if(megaMults > maxMegaMults_NonSimdMul){
            maxMegaMults_NonSimdMul = megaMults;
        }
    }
    
    
    for(t = 0; t < NUMTRIES; t++){
        double time4 = omp_get_wtime();
        float sum1 = SimdMulSum(a,b,len);
        double time5 = omp_get_wtime();
        double megaMults = (double)ARRAYSIZE/(time5-time4)/1000000.;
        if(megaMults > maxMegaMults_SimdMulSum){
            maxMegaMults_SimdMulSum = megaMults;
        }
    }

    
    for(t = 0; t < NUMTRIES; t++){
        double time6 = omp_get_wtime();
        float sum2 = NonSimdMulSum(a,b,len);
        double time7 = omp_get_wtime();
        double megaMults = (double)ARRAYSIZE/(time7-time6)/1000000.;
        if(megaMults > maxMegaMults_NonSimdMulSum){
            maxMegaMults_NonSimdMulSum = megaMults;
        }
    }
    
    
    float mulSpeedUp = 0.0;
    float sumSpeedUp = 0.0;
    
    
    mulSpeedUp = maxMegaMults_SimdMul / maxMegaMults_NonSimdMul;
    
    sumSpeedUp = maxMegaMults_SimdMulSum / maxMegaMults_NonSimdMulSum;
    
    
    printf("%d\t%4.4lf\t%4.4lf\n",ARRAYSIZE,mulSpeedUp,sumSpeedUp);
    
}



void
SimdMul( float *a, float *b,   float *c,   int len )
{
    int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
    __asm
    (
     ".att_syntax\n\t"
     "movq    -24(%rbp), %r8\n\t"        // a
     "movq    -32(%rbp), %rcx\n\t"        // b
     "movq    -40(%rbp), %rdx\n\t"        // c
     );
    
    for( int i = 0; i < limit; i += SSE_WIDTH )
    {
        __asm
        (
         ".att_syntax\n\t"
         "movups    (%r8), %xmm0\n\t"    // load the first sse register
         "movups    (%rcx), %xmm1\n\t"    // load the second sse register
         "mulps    %xmm1, %xmm0\n\t"    // do the multiply
         "movups    %xmm0, (%rdx)\n\t"    // store the result
         "addq $16, %r8\n\t"
         "addq $16, %rcx\n\t"
         "addq $16, %rdx\n\t"
         );
    }
    
    for( int i = limit; i < len; i++ )
    {
        c[i] = a[i] * b[i];
    }
}



float
SimdMulSum( float *a, float *b, int len )
{
    float sum[4] = { 0., 0., 0., 0. };
    int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
    
    __asm
    (
     ".att_syntax\n\t"
     "movq    -40(%rbp), %r8\n\t"        // a
     "movq    -48(%rbp), %rcx\n\t"        // b
     "leaq    -32(%rbp), %rdx\n\t"        // &sum[0]
     "movups     (%rdx), %xmm2\n\t"        // 4 copies of 0. in xmm2
     );
    
    for( int i = 0; i < limit; i += SSE_WIDTH )
    {
        __asm
        (
         ".att_syntax\n\t"
         "movups    (%r8), %xmm0\n\t"    // load the first sse register
         "movups    (%rcx), %xmm1\n\t"    // load the second sse register
         "mulps    %xmm1, %xmm0\n\t"    // do the multiply
         "addps    %xmm0, %xmm2\n\t"    // do the add
         "addq $16, %r8\n\t"
         "addq $16, %rcx\n\t"
         );
    }
    
    __asm
    (
     ".att_syntax\n\t"
     "movups     %xmm2, (%rdx)\n\t"    // copy the sums back to sum[ ]
     );
    
    for( int i = limit; i < len; i++ )
    {
        sum[0] += a[i] * b[i];
    }
    
    return sum[0] + sum[1] + sum[2] + sum[3];
}



float
NonSimdMulSum( float *a, float *b, int len )
{
    float sum[4] = { 0., 0., 0., 0. };
    int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
    
    for( int i = 0; i < len; i++ )
    {
        sum[0] += a[i] * b[i];
    }
    
    return sum[0];
}

void NonSimdMul(float *a, float *b,   float *c,   int len){
    
    for( int i = 0; i < len; i++ )
    {
        c[i] = a[i] * b[i];
    }
    
}
