/*
 * FEM_Grid_Step_PML - C-code based FEM with PML (Perfectly Matched Layer) and
 *                     speed-of-sound compensation for high frequencies
 *
 * To build this with Matlab, run >> dmex FEM_Grid_Step_PML.c
 * This will extract the command line from the following comment line:
 *
 * <<MALTAB-MEX-COMMAND>> mex -R2018a CFLAGS="-Ofast" FEM_Grid_Step_PML.c
 *
 * This is a MEX file for MATLAB. Created July/Aug 2020, D. McGrath
 *
 * Copyright Dolby Laboratories, 2020
 */

#include "mex.h"
#include "matrix.h"
#include <stdint.h>
#include <pthread.h>

// Rev 1.1.0 - first real 'release' Aug-2020
// Rev 1.2.0 - Swapped the order of input/output ops so output is grabbed before input is added
// Rev 1.3.0 - Added the ability to turn off the PML in any of the 6 external faces of the grid
// Rev 1.4.0 - Added the capability for attenuation of pressure nodes
#define VERSION_STR "1.4.0"

#define xp1(s) str(1+s)
#define str(s) #s

#if 1
#define P( x,y,z) ( G[ 0 + 4L * ( (uint64_t) (x) + (nX+1) * (uint64_t) ( (y) + (nY+1) * (z) ) ) ] )
#define Vx(x,y,z) ( G[ 1 + 4L * ( (uint64_t) (x) + (nX+1) * (uint64_t) ( (y) + (nY+1) * (z) ) ) ] )
#define Vy(x,y,z) ( G[ 2 + 4L * ( (uint64_t) (x) + (nX+1) * (uint64_t) ( (y) + (nY+1) * (z) ) ) ] )
#define Vz(x,y,z) ( G[ 3 + 4L * ( (uint64_t) (x) + (nX+1) * (uint64_t) ( (y) + (nY+1) * (z) ) ) ] )
#else
#define P( x,y,z) G[0][x][y][z]
#define Vx(x,y,z) G[1][x][y][z]
#define Vy(x,y,z) G[2][x][y][z]
#define Vz(x,y,z) G[3][x][y][z]
#endif
#define Qx(x,y,z,se) ( Qx2[ (x) + (nPML) * ( (y) + (uint64_t) (nY  ) * ( (z) + (nZ  ) * (se) ) ) ] )
#define Qy(x,y,z,se) ( Qy2[ (x) + (nX  ) * ( (y) + (uint64_t) (nPML) * ( (z) + (nZ  ) * (se) ) ) ] )
#define Qz(x,y,z,se) ( Qz2[ (x) + (nX  ) * ( (y) + (uint64_t) (nY  ) * ( (z) + (nPML) * (se) ) ) ] )
#define wM(x,y,z,n) ( ( wallMask[ (x) + (nX+2) * ( (uint64_t) (y) + (nY+2) * (z) )] >> (n) ) & 1 )
#define PML_Pd(p)  ( PMLCoefs[ 0 + 5*(p) ] )
#define PML_Vd(p)  ( PMLCoefs[ 1 + 5*(p) ] )
#define PML_Vi(p)  ( PMLCoefs[ 2 + 5*(p) ] )
#define PML_Qd(p)  ( PMLCoefs[ 3 + 5*(p) ] )
#define PML_Qi(p)  ( PMLCoefs[ 4 + 5*(p) ] )


#define baseArgsTyped \
float*    __restrict G,        \
uint64_t* __restrict wallInds, \
uint8_t*  __restrict wallMask, \
float*    __restrict Qx2,      \
float*    __restrict Qy2,      \
float*    __restrict Qz2,      \
float*    __restrict PMLCoefs, \
int nX, int nY, int nZ, int nPML, float c, float c1, float ca

#define baseArgs G, wallInds, wallMask, Qx2, Qy2, Qz2, PMLCoefs, nX, nY, nZ, nPML, c, c1, ca

#define allArgsTyped int y, int yPML, int z, int zPML, baseArgsTyped
#define allArgs y, yPML, z, zPML, baseArgs

typedef struct s_args {
    int zMin;
    int zMax;
    mxLogical* PML_en;
    float*    __restrict G;
    uint64_t* __restrict wallInds;
    uint8_t*  __restrict wallMask;
    float*    __restrict Qx2;
    float*    __restrict Qy2;
    float*    __restrict Qz2;
    float*    __restrict PMLCoefs;
    int nX; int nY; int nZ; int nPML; float c; float c1; float ca;
} all_args;

static inline void update_P( int x, int xSE, int xPML, int ySE, int zSE, allArgsTyped ) {
    // (G) Update Q from V.diff
    // (B) Update P from Q, then attenuate Q
    // (C) Update Q from V.diff
    if (xSE>=0) {
        Qx(xPML,y,z,xSE) += PML_Qi(xPML) * (Vx(x+1,y,z) - Vx(x,y,z));
        P(x,y,z)         += PML_Pd(xPML) * Qx(xPML,y,z,xSE);
        Qx(xPML,y,z,xSE) *= PML_Qd(xPML);
        Qx(xPML,y,z,xSE) += PML_Qi(xPML) * (Vx(x+1,y,z) - Vx(x,y,z));
    }
    if (ySE>=0) {
        Qy(x,yPML,z,ySE) += PML_Qi(yPML) * (Vy(x,y+1,z) - Vy(x,y,z));
        P(x,y,z)         += PML_Pd(yPML) * Qy(x,yPML,z,ySE);
        Qy(x,yPML,z,ySE) *= PML_Qd(yPML);
        Qy(x,yPML,z,ySE) += PML_Qi(yPML) * (Vy(x,y+1,z) - Vy(x,y,z));
    }
    if (zSE>=0) {
        Qz(x,y,zPML,zSE) += PML_Qi(zPML) * (Vz(x,y,z+1) - Vz(x,y,z));
        P(x,y,z)         += PML_Pd(zPML) * Qz(x,y,zPML,zSE);
        Qz(x,y,zPML,zSE) *= PML_Qd(zPML);
        Qz(x,y,zPML,zSE) += PML_Qi(zPML) * (Vz(x,y,z+1) - Vz(x,y,z));
    }
    // (A) Update P from V.diff
    P(x,y,z) += c * (Vx(x,y,z) - Vx(x+1,y,z) +
                     Vy(x,y,z) - Vy(x,y+1,z) +
                     Vz(x,y,z) - Vz(x,y,z+1) );
    
    //if ((xSE<0) && (ySE<0) && (zSE<0))  *total_pwr_ptr = std::max(*total_pwr_ptr, fabsf(P(x,y,z)) );
}

static inline void update_Vx( int x, int xSE, int xPML, int ySE, int zSE, allArgsTyped ) {
    // attenuate V
    if (xSE>=0) Vx(x+1,y,z) *= PML_Vd(xPML);
    // (D) Update V from P.diff
    Vx(x+1,y,z) += c1 * ( P(x,y,z) - P(x+1,y,z) );
    if (wM(x-0,y,z,0)) Vx(x+1,y,z) += ca * ( P(x  ,y,z) - P(x-1,y,z) );
    if (wM(x+2,y,z,0)) Vx(x+1,y,z) += ca * ( P(x+2,y,z) - P(x+1,y,z) );
    // (E) Update V from P.diff (PML)
    if (xSE>=0) Vx(x+1,y,z) += PML_Vi(xPML) * -( P(x,y,z) - P(x+1,y,z) );
}

static inline void update_Vy( int x, int xSE, int xPML, int ySE, int zSE, allArgsTyped ) {
    // attenuate V
    if (ySE>=0) Vy(x,y+1,z) *= PML_Vd(yPML);
    // (D) Update V from P.diff
    Vy(x,y+1,z) += c1 * ( P(x,y,z) - P(x,y+1,z) );
    if (wM(x,y-0,z,1)) Vy(x,y+1,z) += ca * ( P(x,y  ,z) - P(x,y-1,z) );
    if (wM(x,y+2,z,1)) Vy(x,y+1,z) += ca * ( P(x,y+2,z) - P(x,y+1,z) );
    // (E) Update V from P.diff (PML)
    if (ySE>=0) Vy(x,y+1,z) += PML_Vi(yPML) * -( P(x,y,z) - P(x,y+1,z) );
}

static inline void update_Vz( int x, int xSE, int xPML, int ySE, int zSE, allArgsTyped ) {
    // attenuate V
    if (zSE>=0) Vz(x,y,z+1) *= PML_Vd(zPML);
    // (D) Update V from P.diff
    Vz(x,y,z+1) += c1 * ( P(x,y,z) - P(x,y,z+1) );
    if (wM(x,y,z-0,2)) Vz(x,y,z+1) += ca * ( P(x,y,z  ) - P(x,y,z-1) );
    if (wM(x,y,z+2,2)) Vz(x,y,z+1) += ca * ( P(x,y,z+2) - P(x,y,z+1) );
    // (E) Update V from P.diff (PML)
    if (zSE>=0) Vz(x,y,z+1) += PML_Vi(zPML) * -( P(x,y,z) - P(x,y,z+1) );
}

static inline void update_P_row( int ySE, int zSE, mxLogical* PML_en, allArgsTyped ) {
    int x=0;
    int xPMLs = PML_en[0] ? 0 : -1;
    int xPMLe = PML_en[3] ? 1 : -1;
    for (; x<nPML   ; x++) update_P( x, xPMLs,      x, ySE, zSE, allArgs);
    for (; x<nX-nPML; x++) update_P( x,    -1,      0, ySE, zSE, allArgs);
    for (; x<nX     ; x++) update_P( x, xPMLe, nX-1-x, ySE, zSE, allArgs);
}

static inline void update_Vx_row( int ySE, int zSE, mxLogical* PML_en, allArgsTyped ) {
    int x;
    int xPMLs = PML_en[0] ? 0 : -1;
    int xPMLe = PML_en[3] ? 1 : -1;
    x=0;
    for (; x<nPML     ; x++) update_Vx( x, xPMLs,      x, ySE, zSE, allArgs);
    for (; x<nX-nPML-1; x++) update_Vx( x,    -1,      1, ySE, zSE, allArgs);
    for (; x<nX-1     ; x++) update_Vx( x, xPMLe, nX-2-x, ySE, zSE, allArgs);
    for (; x<nX       ; x++) update_Vx( x,    -1,      0, ySE, zSE, allArgs);
}

static inline void update_Vy_row( int ySE, int zSE, mxLogical* PML_en, allArgsTyped ) {
    int x;
    int xPMLs = PML_en[0] ? 0 : -1;
    int xPMLe = PML_en[3] ? 1 : -1;
    x=0;
    for (; x<nPML     ; x++) update_Vy( x, xPMLs,      x, ySE, zSE, allArgs);
    for (; x<nX-nPML-1; x++) update_Vy( x,    -1,      1, ySE, zSE, allArgs);
    for (; x<nX-1     ; x++) update_Vy( x, xPMLe, nX-2-x, ySE, zSE, allArgs);
    for (; x<nX       ; x++) update_Vy( x,    -1,      0, ySE, zSE, allArgs);
}

static inline void update_Vz_row( int ySE, int zSE, mxLogical* PML_en, allArgsTyped ) {
    int x;
    int xPMLs = PML_en[0] ? 0 : -1;
    int xPMLe = PML_en[3] ? 1 : -1;
    x=0;
    for (; x<nPML     ; x++) update_Vz( x, xPMLs,      x, ySE, zSE, allArgs);
    for (; x<nX-nPML-1; x++) update_Vz( x,    -1,      1, ySE, zSE, allArgs);
    for (; x<nX-1     ; x++) update_Vz( x, xPMLe, nX-2-x, ySE, zSE, allArgs);
    for (; x<nX       ; x++) update_Vz( x,    -1,      0, ySE, zSE, allArgs);
}

void * run_rows_P(void *tmp_arg) {
    
    all_args* args = (all_args*) tmp_arg;
    int zMin=args->zMin, zMax=args->zMax;
    mxLogical *PML_en = args->PML_en;
    float*    __restrict G = args->G;
    uint64_t* __restrict wallInds = args->wallInds;
    uint8_t*  __restrict wallMask = args->wallMask;
    float*    __restrict Qx2      = args->Qx2;
    float*    __restrict Qy2      = args->Qy2;
    float*    __restrict Qz2      = args->Qz2;
    float*    __restrict PMLCoefs = args->PMLCoefs;
    int nX=args->nX, nY=args->nY, nZ=args->nZ, nPML=args->nPML;
    float c=args->c, c1=args->c1, ca=args->ca;
    
    // Do passes (G),(A),(B),(C)
    // Loop over Z
    for (int z1=zMin; z1<zMax; z1++) {
        
        // Loop over Y
        for (int y1=0; y1<nY; y1++) {
            int z=z1, y=y1;
            
            int zSE=-1, zPML=nPML;
            if ((z<nPML) && (PML_en[2])) { zSE = 0; zPML   = z; }
            else if ((z>nZ-1-nPML) && (PML_en[5])) { zSE = 1; zPML   = nZ-1-z; }

            int ySE=-1, yPML=nPML;
            if ((y<nPML) && (PML_en[1])) { ySE = 0; yPML   = y; }
            else if ((y>nY-1-nPML) && (PML_en[4])) { ySE = 1; yPML   = nY-1-y; }
            
            update_P_row( ySE, zSE, PML_en, allArgs );
            
        }
    }
    return NULL;
}

void * run_rows_V(void *tmp_arg) {
    
    all_args* args = (all_args*) tmp_arg;
    int zMin=args->zMin, zMax=args->zMax;
    mxLogical *PML_en = args->PML_en;
    float*    __restrict G = args->G;
    uint64_t* __restrict wallInds = args->wallInds;
    uint8_t*  __restrict wallMask = args->wallMask;
    float*    __restrict Qx2      = args->Qx2;
    float*    __restrict Qy2      = args->Qy2;
    float*    __restrict Qz2      = args->Qz2;
    float*    __restrict PMLCoefs = args->PMLCoefs;
    int nX=args->nX, nY=args->nY, nZ=args->nZ, nPML=args->nPML;
    float c=args->c, c1=args->c1, ca=args->ca;
    
    // Do passes (D),(E),(F)
    // Loop over Z
    for (int z=zMin; z<zMax; z++) {
        int zSE=-1, zPML=nPML;
        if ((z<nPML) && (PML_en[2])) { zSE = 0; zPML = z; }
        else if ((z>nZ-2-nPML) && (PML_en[5])) { zSE = 1; zPML = nZ-2-z; }
        if (z==nZ-1) zSE=-1;
        
        // Loop over Y
        for (int y=0; y<nY; y++) {
            int ySE=-1, yPML=nPML;
            if ((y<nPML) && (PML_en[1])) { ySE = 0; yPML = y; }
            else if ((y>nY-2-nPML) && (PML_en[4])) { ySE = 1; yPML = nY-2-y; }
            if (y==nY-1) ySE=-1;
            
            update_Vx_row( ySE, zSE, PML_en, allArgs );
            update_Vy_row( ySE, zSE, PML_en, allArgs );

        }
    }
    
    // Do passes (D),(E),(F)
    // Loop over Y
    for (int y=0; y<nY; y++) {
        int ySE=-1, yPML=nPML;
        if ((y<nPML) && (PML_en[1])) { ySE = 0; yPML = y; }
        else if ((y>nY-2-nPML) && (PML_en[4])) { ySE = 1; yPML = nY-2-y; }
        if (y==nY-1) ySE=-1;
        
        // Loop over Z
        for (int z=zMin; z<zMax; z++) {
            int zSE=-1, zPML=nPML;
            if ((z<nPML) && (PML_en[2])) { zSE = 0; zPML = z; }
            else if ((z>nZ-2-nPML) && (PML_en[5])) { zSE = 1; zPML = nZ-2-z; }
            if (z==nZ-1) zSE=-1;
            
            update_Vz_row( ySE, zSE, PML_en, allArgs );
            
        }
    }
    return NULL;
}




// An un-documented function that we need to call, so that we can do
// in-place modification of the GridData safely:
//bool mxUnshareArray(mxArray *array_ptr, bool noDeepCopy);

#define assert(X,msg) if (!(X)) mexErrMsgTxt(msg);
#define myAssert(X,msg) if (!(X)) mexErrMsgTxt(msg);




#define assertType(n,t)     myAssert(mxIsClass(prhs[n], #t), "Expected arg[" xp1(n) "] to be type " #t )
#define assertNDims(n,d)    myAssert(mxGetNumberOfDimensions(prhs[n])==d, "Expected ndims( arg[" xp1(n) "] ) = " #d)
#define assertScalar(n)     myAssert(mxGetNumberOfElements(prhs[n])==1,  "Expected arg[" xp1(n) "] to be be scalar")
#define assertDimLen(n,d,l) myAssert(mxGetDimensions(prhs[n])[d]==l,     "Expected size( arg[" xp1(n) "], " xp1(d) ") = " #l)


void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Usage :
    // [ ...
    //  0    OutSamples,  ... float  [ nSamp x nOut ]
    //  1    maxPwr,      ... float  scalar
    //  2    G,           ... float  [ 4 x NX+1 x NY+1 x NZ+1 ]
    //  3    Qx,          ... float  [ nPML x NY x NZ x 2 ]
    //  4    Qy,          ... float  [ NX x nPML x NZ x 2 ]
    //  5    Qz,          ... float  [ NY x NZ x nPML x 2 ]
    // ] = ...
    // FEM_Grid_Step( ...
    //  ...
    //  0    G,           ... float [ 4 x NX+1 x NY+1 x NZ+1 ]
    //  1    StepRate,    ... float scalar
    //  2    StepAlpha,   ... float scalar
    //  3    WallIndices, ... uint64 [ nWalls x 1]
    //  4    InIndex,     ... uint64 [ 1 x nIn ]
    //  5    InSample,    ... float  [ nSamp x nIn ]
    //  6    WallMask,    ... uint8  [ NX+1 x NY+1 x NZ+1 ]
    //  7    Qx,          ... float  [ nPML x NY x NZ x 2 ]
    //  8    Qy,          ... float  [ NX x nPML x NZ x 2 ]
    //  9    Qz,          ... float  [ NY x NZ x nPML x 2 ]
    // 10    PMLCoefs,    ... float  [ nPML x 5 ]
    // 11    OutIndices   ... uint64 [ nOut x 1 ]
    // 12    PML_Enable   ... logical[ 3 x 2 ]
    // 13    PLossInds    ... uint64 [ nPLoss x 1 ]
    //  ...
    //      );
    
    // Report versionn number
    if (nrhs==0) {
        myAssert(nlhs <= 1,  "Expected 1 output Arg for version string");
        plhs[0] = mxCreateString( VERSION_STR );
        return;
    }
    
    myAssert(nrhs == 14, "Expected 14 input Args");
    myAssert(nlhs == 1,  "Expected 1 output Args");
    
    // Get a few key constants from the array sizes:
    uint64_t nX     = mxGetDimensions(prhs[0 ])[1]-1;
    uint64_t nY     = mxGetDimensions(prhs[0 ])[2]-1;
    uint64_t nZ     = mxGetDimensions(prhs[0 ])[3]-1;
    uint64_t nWalls = mxGetDimensions(prhs[3 ])[0];
    uint64_t nIn    = mxGetDimensions(prhs[4 ])[1];
    uint64_t nOut   = mxGetDimensions(prhs[11])[1];
    uint64_t nPML   = mxGetDimensions(prhs[10])[1];
    uint64_t nSamp  = mxGetDimensions(prhs[5 ])[0];
    uint64_t nPLoss = mxGetDimensions(prhs[13])[0];
    
    // Check all the types with assertions
    assertType(0, single); assertNDims(0, 4); assertDimLen(0, 0,4);
    assertType(1, single); assertNDims(1, 2); assertScalar(1 );
    assertType(2, single); assertNDims(2, 2); assertScalar(2 );
    assertType(3, uint64); assertNDims(3, 2); assertDimLen(3, 0,nWalls); assertDimLen(3, 1,1);
    assertType(4, uint64); assertNDims(4, 2); assertDimLen(4, 0,1);      assertDimLen(4, 1,nIn);
    assertType(5, single); assertNDims(5, 2); assertDimLen(5, 0,nSamp);  assertDimLen(5, 1,nIn);
    assertType(6, uint8 ); assertNDims(6, 3); assertDimLen(6, 0,nX+2);   assertDimLen(6, 1,nY+2); assertDimLen(6,2,nZ+2);
    assertType(7, single); assertNDims(7, 4); assertDimLen(7, 0,nPML);   assertDimLen(7, 1,nY);   assertDimLen(7,2,nZ);   assertDimLen(7,3,2);
    assertType(8, single); assertNDims(8, 4); assertDimLen(8, 0,nX);     assertDimLen(8, 1,nPML); assertDimLen(8,2,nZ);   assertDimLen(8,3,2);
    assertType(9, single); assertNDims(9, 4); assertDimLen(9, 0,nX);     assertDimLen(9, 1,nY);   assertDimLen(9,2,nPML); assertDimLen(9,3,2);
    assertType(10,single); assertNDims(10,2); assertDimLen(10,0,5);      assertDimLen(10,1,nPML);
    assertType(11,uint64); assertNDims(11,2); assertDimLen(11,0,1);      assertDimLen(11,1,nOut);
    assertType(12,logical);assertNDims(12,2); assertDimLen(12,0,3);      assertDimLen(12,1,2);
    assertType(13,uint64); assertNDims(13,2); assertDimLen(13,0,nPLoss); assertDimLen(13,1,1);

    float c = mxGetSingles(prhs[1])[0];
    float a = mxGetSingles(prhs[2])[0];

    float* G   = mxGetSingles(prhs[0]);
    
    plhs[0] = mxCreateNumericMatrix( nSamp, nOut, mxSINGLE_CLASS, mxREAL);
    float * outSig = mxGetSingles(plhs[0]);

    uint64_t* wallInds = mxGetUint64s(prhs[ 3]);
    uint64_t* inInds   = mxGetUint64s(prhs[ 4]);
    float*    inSig    = mxGetSingles(prhs[ 5]);
    uint8_t*  wallMask = mxGetUint8s (prhs[ 6]);
    float*    Qx2      = mxGetSingles(prhs[ 7]);
    float*    Qy2      = mxGetSingles(prhs[ 8]);
    float*    Qz2      = mxGetSingles(prhs[ 9]);
    float*    PMLCoefs = mxGetSingles(prhs[10]);
    uint64_t* outInds  = mxGetUint64s(prhs[11]);
    mxLogical* PML_en  = mxGetLogicals(prhs[12]);
    uint64_t* pLossInds= mxGetUint64s(prhs[13]);

    float c1 = c*(1+2*a);
    float ca = c*a;
    
    for (int sample_num=0; sample_num<nSamp; sample_num++) {
        
        // Apply pressure-loss, to emulate abosrbing wall
        for(int p=0; p<nPLoss; p++) G[ pLossInds[p] -1 ] *= (1.0f-c);

        // Get Output samples
        for(int n=0; n<nOut; n++) outSig[ sample_num + nSamp*n ] = G[ outInds[n] -1 ];
        // Add input samples
        for(int n=0; n<nIn;  n++) G[ inInds[n] -1 ] += inSig[ sample_num + nSamp*n ];
        
        //
        // -----(N)----------(N+0.5)-----------(N+1)---------(N+1.5)-> time
        //
        // -----(A')--+---+--->[#]--------------------------->[#]---->   P
        //           /   /        \
        //         (A) (B)       (D,E)
        //         /   /            \
        // ---->[#]---/---------(C)--+-(F)----->[#]------------------>   V
        //           /      \             \
        //          /       (C)           (G)
        //         /          \             \
        // ---->[#]-------(C)--+-------------+->[#]------------------>   Q
        //       ^                               ^
        //       |                               |
        // State at start of Step()        State at end of Step()
        //
        // We will implement one time step using the following operations:
        //
        // ### (in) Inject the input signal (usually into P)
        // ### (out) record the output signal (usually from P)
        // ### (A') Apply Pressure-loss attenuation to some cells of P
        // ### (A) Update P from V.diffX (normal wave equation)
        // ### (B) Update P from Q       (PML)
        // ### (C) Update Q from V.diffX (PML) (incl attenuation of Q & V)
        // ### (D) Update V from P.diffX (normal wave equation)
        // ### (E) Update V from P.diffX (PML) (additional amount)
        // ### (F) Force wall values     (normal wave equation boundaries)
        // ### (G) Update Q from V.diffX (PML)
        //

#define maxExtraThreads 7
        pthread_t thread_id[maxExtraThreads];
        all_args args[maxExtraThreads+1];

        uint64_t total_size = (uint64_t)nX * (uint64_t)nY * (uint64_t)nZ ;
        uint64_t numExtraThreads = total_size/(uint64_t)100000;
        if (numExtraThreads > maxExtraThreads) numExtraThreads = maxExtraThreads;
        //numExtraThreads = 0;
        
        int zMin;
        
        zMin = 0;
        for (int t=0; t<numExtraThreads; t++) {
            int zMax = (nZ*(t+1))/(numExtraThreads+1);
            args[t] = (all_args){zMin, zMax, PML_en, baseArgs};
            pthread_create(&thread_id[t], NULL, &run_rows_P, &args[t]);
            zMin = zMax;
        }
        args[numExtraThreads] = (all_args){zMin, nZ, PML_en, baseArgs};
        run_rows_P( &args[numExtraThreads] );
        for (int t=0; t<numExtraThreads; t++) {
            void *val;
            pthread_join( thread_id[t], &val);
        }
        
        zMin = 0;
        for (int t=0; t<numExtraThreads; t++) {
            int zMax = (nZ*(t+1))/(numExtraThreads+1);
            args[t] = (all_args){zMin, zMax, PML_en, baseArgs};
            pthread_create(&thread_id[t], NULL, &run_rows_V, &args[t]);
            zMin = zMax;
        }
        args[numExtraThreads] = (all_args){zMin, nZ, PML_en, baseArgs};
        run_rows_V( &args[numExtraThreads] );
        for (int t=0; t<numExtraThreads; t++) {
            void *val;
            pthread_join( thread_id[t], &val);
        }
        
        // Force walls to zero
        for(int w=0; w<nWalls; w++) G[ wallInds[w] -1 ] = 0.0f;
        
    }
    
}

