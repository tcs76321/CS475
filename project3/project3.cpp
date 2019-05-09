#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cstdio>

#ifndef _OPENMP
fprintf( stderr, "No OpenMP support!\n" );
return 1;
#endif


float SQR( float x )
{
    return x*x;
}

omp_lock_t    Lock;
int        NumInThreadTeam;
int        NumAtBarrier;
int        NumGone;

void InitBarrier( int n )
{
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock( &Lock );
}

void WaitBarrier( )
{
    omp_set_lock( &Lock );
    {
        NumAtBarrier++;
        if( NumAtBarrier == NumInThreadTeam )
        {
            NumGone = 0;
            NumAtBarrier = 0;
            // let all other threads get back to what they were doing
            // before this one unlocks, knowing that they might immediately
            // call WaitBarrier( ) again:
            while( NumGone != NumInThreadTeam-1 );
            omp_unset_lock( &Lock );
            return;
        }
    }
    omp_unset_lock( &Lock );
    
    while( NumAtBarrier != 0 );    // this waits for the nth thread to arrive
    
    #pragma omp atomic
    NumGone++;            // this flags how many threads have returned
}

unsigned int seed = 0;

int    NowYear;        // 2019 - 2024
int    NowMonth;        // 0 - 11

float    NowPrecip;        // inches of rain per month
float    NowTemp;        // temperature this month
float    NowHeight;        // grain height in inches
int    NowNumDeer;        // number of deer in the current population

int NowNumWolves;
int WolvesStarve;


const float GRAIN_GROWS_PER_MONTH =        16.0; // in inches
const float ONE_DEER_EATS_PER_MONTH =        0.5;

const float AVG_PRECIP_PER_MONTH =        6.0;    // average - in inches as well
const float AMP_PRECIP_PER_MONTH =        6.0;    // plus or minus
const float RANDOM_PRECIP =            2.0;    // plus or minus noise

const float AVG_TEMP =                50.0;    // average - in F degrees
const float AMP_TEMP =                20.0;    // plus or minus
const float RANDOM_TEMP =            10.0;    // plus or minus noise

const float MIDTEMP =                40.0;
const float MIDPRECIP =                10.0;



float
Ranf( unsigned int *seedp,  float low, float high )
{
    float r = (float) rand_r( seedp );              // 0 - RAND_MAX
    
    return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int
Ranf( unsigned int *seedp, int ilow, int ihigh )
{
    float low = (float)ilow;
    float high = (float)ihigh + 0.9999f;
    
    return (int)(  Ranf(seedp, low,high) );
}

void CalcEnvironment(){
    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
    
    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );
    
    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. )
        NowPrecip = 0.;
}

void PrintEnvironment(){
    printf("%4.4lf\t%4.4lf\t%4.4lf\t%d\t%d\n",NowTemp, NowPrecip, NowHeight, NowNumDeer, NowNumWolves);
}


void GrainDeer(){
    
    while( NowYear < 2025 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        int NextNumDeer = NowNumDeer + NowNumDeer/2 + 1;
        
        if(NowHeight < (NextNumDeer*ONE_DEER_EATS_PER_MONTH)){
            NextNumDeer = (NextNumDeer/3) + 1;
        }
        
        NextNumDeer = NextNumDeer - NowNumWolves/2;
        
        if(NextNumDeer < 0){
            NextNumDeer = 2;
        }
        //NextNumDeer = NextNumDeer - (NowNumWolves/4);
        
        
        
        // DoneComputing barrier:
        WaitBarrier();
        
        NowNumDeer = NextNumDeer;
        // DoneAssigning barrier:
        WaitBarrier();
        
        // DonePrinting barrier:
        WaitBarrier();
        
    }
    
}

void Grain(){
    
    while( NowYear < 2025 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        int NewHeight = NowHeight;
        
        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
        
        NewHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        NewHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
        
        if(NewHeight < 0){
            NewHeight = 0;
        }
        
        // DoneComputing barrier:
        WaitBarrier();
        
        NowHeight = NewHeight;
        // DoneAssigning barrier:
        WaitBarrier();
        
        
        // DonePrinting barrier:
        WaitBarrier();
    }
    
}

void Watcher(){
    
    while( NowYear < 2025 )
    {
        //NOTHING
        // DoneComputing barrier:
        WaitBarrier();
        
        //NOTHING
        // DoneAssigning barrier:
        WaitBarrier();
        
        
        //PRINT RESULTS AND INCREMENT TIME
        
        //CALCULATE NEW ENVIRO PARAMS
        
        NowMonth = NowMonth + 1;
        if(NowMonth > 11){
            NowMonth = 0;
            NowYear = NowYear +1;
        }
        
        CalcEnvironment();
        PrintEnvironment();
        
        // DonePrinting barrier:
        WaitBarrier();
        
    }
    
}

void Wolves(){// AKA my agent
    
    while( NowYear < 2025 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        int NextNumWolves = NowNumWolves + NowNumWolves/4 + 1;
        
        if(NextNumWolves*2 > NowNumDeer){
            NextNumWolves = NowNumWolves - 2;
        }
        
        if(NextNumWolves < 1){
            NextNumWolves = 1;
        }
        
        
        // DoneComputing barrier:
        WaitBarrier();
        
        NowNumWolves = NextNumWolves;
        // DoneAssigning barrier:
        WaitBarrier();
        
        
        // DonePrinting barrier:
        WaitBarrier();
        
    }
    
}


int main(){
    
    // starting date and time:
    NowMonth =    0;
    NowYear  = 2019;
    
    // starting state (feel free to change this if you want):
    NowNumDeer = 5;
    NowHeight =  3.;
    
    NowNumWolves = 2;
    WolvesStarve = 0;
    
    CalcEnvironment();
    PrintEnvironment();
    
    InitBarrier(4);
    
    omp_set_num_threads( 4 );    // same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            GrainDeer( );
        }
        
        #pragma omp section
        {
            Grain( );
        }
        
        #pragma omp section
        {
            Watcher( );
        }
        
        #pragma omp section
        {
            Wolves( );    // your own
        }
    }
}
