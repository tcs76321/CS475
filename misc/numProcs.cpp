#include <stdio.h>
#include <omp.h>

int main(){

	int num;
	num = omp_get_num_procs();
	printf("\nThe number of procs/cores/hyperthreads on this computer is %d\n\n", num);

}
