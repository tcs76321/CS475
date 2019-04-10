/* Trevor Stahl 

Project 0

CS475

Simple OpenMP project to get started

CITATION: web.engr.oregonstate.edu/~mjb/cs575/Projects/proj00.html

minor style changes as I transcribed it 

and additions for precision and average performace and total exe time
*/

#include <omp.h>
#include <stdio.h>
#include <math.h>

#define NUMT 8
#define ARRAYSIZE 200000 // maybe change after tsting if too big or too small
#define NUMTRIES 10 // ""

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

int main(){
#ifndef _OPENMP
        fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
        return 1;
#endif

        double prec = omp_get_wtick();
        printf("Precision is: %12.12lf \n");

        omp_set_num_threads(NUMT);
        fprintf(stderr, "Using %d threads\n", NUMT);

        double maxMegaMults = 0.0;

        double avgMegaMults = 0.0;

        double avgTime = 0.0;

        int t;
        for(t = 0; t < NUMTRIES; t++){
                double time0 = omp_get_wtime();

                #pragma omp parallel for
                for(int i = 0; i < ARRAYSIZE; i++){
                        C[i] = A[i] * B[i];
                }

                double time1 = omp_get_wtime();
                double megaMults = (double)ARRAYSIZE/(time1-time0)/1000000.0;

                if(megaMults > maxMegaMults){
                        maxMegaMults = megaMults;
                }

                avgTime = avgTime + (time1-time0)/NUMTRIES;

                avgMegaMults += megaMults/NUMTRIES;
        }

        printf("Peak Performance = %8.2lf MegaMults per Second\n", maxMegaMults);


        // note: %lf stands for long float which is how printf prints a double
        //       %d stands for "decimal integer"


        printf("Average Performance = %8.2lf MegaMults per Second\n", avgMegaMults);


        printf("Average Total Execution Time: %12.12lf\n", avgTime);

        return 0;
}
