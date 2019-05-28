kernel
void
ArrayMult( global const float *dA, global const float *dB, global const float *dC, global float *dD )
{
    int gid = get_global_id( 0 );
    
    dD[gid] = dA[gid] * dB[gid] + dC[gid];
}
