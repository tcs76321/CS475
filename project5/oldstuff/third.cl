kernel
int
ArrayMult( global const float *dA, global const float *dB, global float *dC )
{
	int gid = get_global_id( 0 );
    
    int res = 0;

    res = dA[gid] * dB[gid];

    return res;
}
