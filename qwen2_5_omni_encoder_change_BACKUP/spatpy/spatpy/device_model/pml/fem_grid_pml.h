#ifndef FEM_GRID_PML
#define FEM_GRID_PML

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <stdint.h>

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

void read_bin(const char *filename, size_t size, void **x, uint64_t *len);
void write_bin(const char *filename, size_t size, const void *x, uint64_t len);

void fem_grid_pml_step(
    float    *outSig,
    float    *G,
    float    stepRate,
    float    stepAlpha,
    uint64_t *wallInds,
    uint64_t *inInds,  
    float    *inSig,   
    uint8_t  *wallMask,
    float    *Qx2,     
    float    *Qy2,     
    float    *Qz2,
    float    *PMLCoefs,
    uint64_t *outInds,
    uint8_t  *PML_en, 
    uint64_t *pLossInds,
    uint64_t nX,    
    uint64_t nY,    
    uint64_t nZ,    
    uint64_t nWalls,
    uint64_t nIn,   
    uint64_t nOut,  
    uint64_t nPML,  
    uint64_t nSamp, 
    uint64_t nPLoss
);

#ifdef __cplusplus
}
#endif

#endif /* FEM_GRID_PML */
