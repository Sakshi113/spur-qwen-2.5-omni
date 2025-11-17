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

#include "fem_grid_pml.h"
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <assert.h>

// #define MALFEASANCE_LOG_ARGS 1

void read_bin(const char *filename, size_t size, void **x, uint64_t *len)
{
    FILE *fp = fopen(filename, "rb");
    fseek(fp, 0L, SEEK_END);
    size_t nbytes = ftell(fp);
    rewind(fp);
    if (len) {
        *len = nbytes / size;
    }
    if (x) {
        *x = malloc(nbytes * size);
        (void) fread(*x, size, nbytes / size, fp);
    }
    fclose(fp);
}

void write_bin(const char *filename, size_t size, const void *x, uint64_t len)
{
    FILE *fp = fopen(filename, "wb");
    fwrite(x, size, len, fp);
    fclose(fp);
}

#define xp1(s) str(1+s)
#define str(s) #s

#define DEBUG_LOGGING 1
#ifdef DEBUG_LOGGING
#define DEBUG_LINE printf("%s:%d\n", __FILE__, __LINE__);
#define DEBUG_XYZ_INDEX(name,x,y,z) printf("%s:%d %s(%d,%d,%d)\n", __FILE__, __LINE__, str(name), x, y, z);
#else
#define DEBUG_LINE
#define DEBUG_XYZ_INDEX(name,x,y,z)
#endif

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

typedef uint8_t mxLogical;

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
)
{
#ifdef MALFEASANCE_LOG_ARGS
    uint64_t sizes[9] = {
        nX,
        nY,
        nZ,
        nWalls,
        nIn,
        nOut,
        nPML,
        nSamp,
        nPLoss
    };

    write_bin("sizes.bin", sizeof(uint64_t), sizes, sizeof(sizes) / sizeof(sizes[0]));
    write_bin("inSig.bin", sizeof(float), inSig, nIn);
    write_bin("G.bin", sizeof(float), G, (nX + 1) * (nY + 1) * (nZ + 1) * 4);
    write_bin("wallInds.bin", sizeof(uint64_t), wallInds, nWalls);
    write_bin("inInds.bin", sizeof(uint64_t), inInds, nIn);
    write_bin("outInds.bin", sizeof(uint64_t), outInds, nOut);
    write_bin("pLossInds.bin", sizeof(uint64_t), pLossInds, nPLoss);
    write_bin("Qx2.bin", sizeof(float), Qx2, 2 * nPML * nY * nZ);
    write_bin("Qy2.bin", sizeof(float), Qy2, 2 * nPML * nX * nZ);
    write_bin("Qz2.bin", sizeof(float), Qz2, 2 * nPML * nX * nY);
    write_bin("PMLCoefs.bin", sizeof(float), PMLCoefs, nPML * 5);
    write_bin("wallMask.bin", sizeof(uint8_t), wallMask, (nX + 2) * (nY + 2) * (nZ + 2));
    write_bin("PML_en.bin", sizeof(uint8_t), PML_en, 6);
#endif

    float c = stepRate;
    float a = stepAlpha;

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
        // numExtraThreads = 0;
        
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
