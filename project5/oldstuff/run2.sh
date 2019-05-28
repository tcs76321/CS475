#!/bin/bash  
for g in 1000 1024
# 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8000000  
do  
    for l in 8 16
# 32 64 128 256 512  
    do  
         g++-8 -DNMB=$g -DLOCAL_SIZE=$l -o first first.cpp -framework OpenCL -lm -fopenmp -lm -fopenmp  
        ./first  
    done  
done
