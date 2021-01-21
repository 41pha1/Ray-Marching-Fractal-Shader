void rotateX(inout vec3 z, float a)
{
	float s = sin(a);
	float c = cos(a);
	z.y = c*z.y + s*z.z;
	z.z = c*z.z - s*z.y;
}

void rotateY(inout vec3 z, float a)
{
	float s = sin(a);
	float c = cos(a);
	z.z = c*z.z + s*z.x;
	z.x = c*z.x - s*z.z;
}

void rotateZ(inout vec3 z, float a)
{
	float s = sin(a);
	float c = cos(a);
	z.x = c*z.x + s*z.y;
	z.y = c*z.y - s*z.x;
}

void distort(inout vec3 z, float s)
{
	float nx, ny, nz;
	float y = z.y;
	ny = sin(z.z);
	nz = z.x*0.9;
	nx = y;
	z = mix(z, vec3(ny, nz, nx), s);
	//z = max(z,-z);
}

void shrink(inout vec3 z, float shrink)
{
	z = z*shrink;
}

void scaleAround(inout vec3 z, vec3 offset, float scale)
{
	z = z*scale - offset*(scale-1.0);
}

void planeFold(inout vec3 z, vec3 n, float d) {
	z.xyz -= 2.0 * min(0.0, dot(z.xyz, n) - d) * n;
}

void sphereFold(inout vec3 z, float minR, float maxR) {
	float r2 = dot(z.xyz, z.xyz);
	z *= max(maxR / max(minR, r2), 1.0);
}

void boxFold(inout vec3 z, vec3 r) {
	z.xyz = clamp(z.xyz, -r, r) * 2.0 - z.xyz;
}

void absFold(inout vec3 z, vec3 c) {
	z.xyz = abs(z.xyz - c) + c;
}
void sierpinskiFold(inout vec3 z) {
	z.xy -= min(z.x + z.y, 0.0);
	z.xz -= min(z.x + z.z, 0.0);
	z.yz -= min(z.y + z.z, 0.0);
}
void mengerFold(inout vec3 z) {
	float a = min(z.x - z.y, 0.0);
	z.x -= a;
	z.y += a;
	a = min(z.x - z.z, 0.0);
	z.x -= a;
	z.z += a;
	a = min(z.y - z.z, 0.0);
	z.y -= a;
	z.z += a;
}

vec3 getSkyColor(vec3 e) {
    e.y = (max(e.y,0.0)*0.8+0.2)*0.8;
    return vec3(pow(1.0-e.y,2.0), 1.0-e.y, 0.6+(1.0-e.y)*0.4) * 1.1;
}

float diffuse(vec3 n,vec3 l,float p) {
    return pow(dot(n,l) * 0.4 + 0.6,p);
}
float specular(vec3 n,vec3 l,vec3 e,float s) {
    float nrm = (s + 8.0) / (3.141 * 8.0);
    return pow(max(dot(reflect(e,n),l),0.0),s) * nrm;
}

const float ITERATIONS = 17.;
const float EPSILON = 0.0001;

float map(vec3 z)
{
    shrink(z, 0.1);
    planeFold(z, vec3(-1.0, 0.0, 0.0), -1.0);

    for(float n = 0.; n < ITERATIONS; n++)
    {
        sphereFold(z, 0.1, 1.2);
        planeFold(z, vec3(0.0, 1.0, 0.0), -1.0);
        planeFold(z, vec3(0.0, 0.0, 1.0), -1.0);
        mengerFold(z);
        scaleAround(z, vec3(1.0, 1.0, 1.0), 2.0);
        rotateX(z,-0.1+sin(iGlobalTime*0.4)*0.1);
        rotateZ(z,-0.048+sin(iGlobalTime*0.77)*0.1);
        //rotateY(z, sin(iGlobalTime*0.2)*0.05);
        sphereFold(z, 0.2, 1.7);
   }
   return (length(z)) * pow(2., -ITERATIONS);
}

vec3 calcNormal(vec3 pos)
{
    vec2 e = vec2(EPSILON, 0.);
    return normalize(vec3(map(pos+e.xyy)-map(pos-e.xyy),map(pos+e.yxy)-map(pos-e.yxy),map(pos+e.yyx)-map(pos-e.yyx)));
}

const float PI = 3.141;
const float PHI = 1.618;

float hash1(float f)
{
  return fract(sin(f*234.345)*83457.32);
}

vec3 nextVec( float i, float n)
{
    float phi = 2.0*PI*fract(i/PHI);
    float zi = 1.0 - (2.0*i+1.0)/n;
    float sinTheta = sqrt( 1.0 - zi*zi);
    return vec3( cos(phi)*sinTheta, sin(phi)*sinTheta, zi);
}

float calcAO( in vec3 pos, in vec3 nor, float pre, float dis)
{
	float ao = 0.0;
    for( int i=0; i<int(pre); i++ )
    {
        vec3 ap = nextVec( float(i), pre );
		ap *= sign( dot(ap,nor) ) * hash1(float(i));
        ao += clamp( map( pos + nor*0.05 + ap*dis)*(32.0/dis), 0.0, 1.0 );
    }
	ao /= pre;

    return clamp( ao*ao, 0.0, 1.0 );
}


float march(vec3 ro, vec3 rd)
{
    float disO = 0.;

    for(int i = 0; i < 100000; i++)
    {
    	vec3 pos = ro+rd*disO;
        float d = map(pos);
        disO += d;

        if(d < EPSILON)
            return disO;
        if(disO > 200.)
            break;
    }
    return -1.;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float AA = 5.;

    vec3 tot = vec3(0.);

    for(float ax = 0.; ax < AA; ax++)
    {
        for(float ay = 0.; ay < AA; ay++)
        {
			vec2 uv = ((fragCoord-0.5*iResolution.xy)/iResolution.y)+vec2(ax / (iResolution.x*AA), ay / (iResolution.y*AA));
           // uv = abs(uv);

            float w = 0.;
            float dis = iGlobalTime-38.;
            vec3 ro = vec3(10., dis, dis);

            vec3 ta = ro+vec3(0., 0.785398, 0.785398);
            vec3 f = normalize(ta-ro);
            vec3 r = normalize(cross(vec3(0.,1., 0.), f));
            vec3 u = normalize(cross(f, r));
            float zoom = 1.;
            vec3 rd = normalize(f*zoom+r*uv.x+u*uv.y);

            vec3 col = vec3(0.);

            float d = march(ro, rd);
            if(d > 0.)
            {
                vec3 pos = ro+rd*d;
                vec3 nor = calcNormal(pos);
                vec3 lDir = normalize(u);
                vec3 base = vec3(0.9);
                float dif = diffuse(nor, lDir, 1.);
                float ao1 = calcAO(pos, nor, 64., 1.5);
                float ao2 = calcAO(pos, nor, 32.,EPSILON * 50.);
                vec3 ref = normalize(lDir - (2.*dot(nor, lDir)*nor));
                float spec = clamp(dot(normalize(pos-ro), ref), 0., 1.);
                spec = pow(spec, 30.);

                vec3 mat = mix(vec3(2., 0.4, 0.3), vec3(0.9), ao2*ao2*ao2);

                col = vec3(mat*ao1*mix(dif, 1., 0.2)*0.8);
                col += spec*0.1;
                col *= 2.6*exp(d*-0.07);
            }
            tot+=col;
        }
    }
    tot /= AA*AA;
    tot = pow( tot, vec3(1.0,1.0,1.4) ) + vec3(0.0,0.02,0.12);
    vec2 q = fragCoord/iResolution.xy;
    tot *= pow( 16.0*q.x*q.y*(1.0-q.x)*(1.0-q.y), 0.2 );
    fragColor =vec4(pow(tot,vec3(0.75)), 1.0);
}
