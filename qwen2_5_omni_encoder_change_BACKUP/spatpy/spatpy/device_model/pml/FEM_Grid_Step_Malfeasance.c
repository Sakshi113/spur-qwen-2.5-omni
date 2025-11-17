/*
 * FEM_Grid_Step_Malfeasance - C-code based FEM with PML (Perfectly Matched Layer) and
 *                     speed-of-sound compensation for high frequencies
 *
 * To build this with Matlab, run >> dmex FEM_Grid_Step_Malfeasance.c
 * This will extract the command line from the following comment line:
 *
 * <<MALTAB-MEX-COMMAND>> mex -R2018a CFLAGS="-Ofast -fvectorize" FEM_Grid_Step_Malfeasance.c fem_grid_pml.c
 *
 * This is a MEX file for MATLAB. Created May 2022, R. Katekar, stolen from FEM_Grid_Step_PML.c by D. McGrath
 *
 * Copyright Dolby Laboratories, 2022
 */

#include "mex.h"
#include "matrix.h"
#include "fem_grid_pml.h"
#include <stdint.h>

#define VERSION_STR "0.0.1"

#define xp1(s) str(1+s)
#define str(s) #s

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

    fem_grid_pml_step(
        outSig,
        G,
        c,
        a,
        wallInds,
        inInds,
        inSig,
        wallMask,
        Qx2,     
        Qy2,     
        Qz2,
        PMLCoefs,
        outInds,
        PML_en, 
        pLossInds,
        nX,    
        nY,    
        nZ,    
        nWalls,
        nIn,   
        nOut,  
        nPML,  
        nSamp, 
        nPLoss
    );
}

