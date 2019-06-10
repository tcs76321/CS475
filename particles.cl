typedef float4 point;
typedef float4 vector;
typedef float4 color;
typedef float4 sphere;


vector
Bounce( vector in, vector n )
{
	vector out = in - n*(vector)( 2.*dot(in.xyz, n.xyz) );
	out.w = 0.;
	return out;
}

vector
BounceSphere( point p, vector v, sphere s )
{
	vector n;
	n.xyz = fast_normalize( p.xyz - s.xyz );
	n.w = 0.;
	return Bounce( v, n );
}

bool
IsInsideSphere( point p, sphere s )
{
	float r = fast_length( p.xyz - s.xyz );
	return  ( r < s.w );
}

kernel
void
Particle( global point *dPobj, global vector *dVel, global color *dCobj )
{
	const float4 G       = (float4) ( 0., -9.8, 0., 0. );
	const float  DT      = 0.01;
	const sphere Sphere1 = (sphere)( -700., -800., 0.,  600. );
	const sphere Sphere2 = (sphere)( 700.0, -800.0, 0.0, 600.0);

	int gid = get_global_id( 0 );

	point  p = dPobj[gid];
	vector v = dVel[gid];
	color  c = dCobj[gid];

	point  pp = p + v * DT + G * (point)(.5*DT*DT);	// p'
	vector vp = v + G * DT;				// v'
	color cp = c;
	pp.w = 1.;
	vp.w = 0.;


	if (IsInsideSphere(pp, Sphere1))
	{
		vp = BounceSphere(p, v, Sphere1);
		pp = p + vp * DT + G * (point)(.5*DT*DT);
		cp = (color)(0., 1., 0., 1.);
	}
	else if (IsInsideSphere(pp, Sphere2)) {
		vp = BounceSphere(p, v, Sphere2);
		pp = p + vp * DT + G * (point)(0.5*DT*DT);
		cp = (color)(1., 0., 0., 1.);
	}


	dPobj[gid] = pp;
	dVel[gid] = vp;
	dCobj[gid] = cp;

}
