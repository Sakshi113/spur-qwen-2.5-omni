#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "torch_cuarrays.h"

#define P( x,y,z) (G[0][x][y][z])
#define Vx(x,y,z) (G[1][x][y][z])
#define Vy(x,y,z) (G[2][x][y][z])
#define Vz(x,y,z) (G[3][x][y][z])
#define wM(x,y,z,n) ((wallMask[x][y][z] >> (n)) & 1)
#define PML_Pd(p)  (PMLCoefs[0][p])
#define PML_Vd(p)  (PMLCoefs[1][p])
#define PML_Vi(p)  (PMLCoefs[2][p])
#define PML_Qd(p)  (PMLCoefs[3][p])
#define PML_Qi(p)  (PMLCoefs[4][p])

template <typename X, typename I>
__global__ void
run_rows_P(
    X stepRate,
    X stepAlpha,
    CuArray<X, 4> G,
    CuArray<X, 4> Qx,
    CuArray<X, 4> Qy,
    CuArray<X, 4> Qz,
    const CuArray<X, 2> PMLCoefs)
{
    I nPML = PMLCoefs.size(1);
    I nX = G.size(1) - 1;
    I nY = G.size(2) - 1;
    I nZ = G.size(3) - 1;
    
    X c = stepRate;
    X a = stepAlpha;
    X c1 = c*(1+2*a);
    X ca = c*a;
    
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int z = threadIdx.x;

    if (x >= nX || y >= nY || z >= nZ) {
        return;
    }

    int zSE=-1, zPML=nPML;
    if (z<nPML) { zSE = 0; zPML   = z; }
    else if (z>nZ-1-nPML) { zSE = 1; zPML   = nZ-1-z; }

    int ySE=-1, yPML=nPML;
    if (y<nPML) { ySE = 0; yPML   = y; }
    else if (y>nY-1-nPML) { ySE = 1; yPML   = nY-1-y; }

    int xPMLs = 0;
    int xPMLe = 1;
    int xSE;
    int xPML;
    if (x < nPML) {
        xSE = xPMLs;
        xPML = x;
    } else if (x < nX - nPML) {
        xSE = -1;
        xPML = 0;
    } else {
        xSE = xPMLe;
        xPML = nX - 1 - x;
    }

    // (G) Update Q from V.diff
    // (B) Update P from Q, then attenuate Q
    // (C) Update Q from V.diff
    if (xSE>=0) {
        Qx[xPML][y][z][xSE] += PML_Qi(xPML) * (Vx(x+1,y,z) - Vx(x,y,z));
        P(x,y,z)         += PML_Pd(xPML) * Qx[xPML][y][z][xSE];
        Qx[xPML][y][z][xSE] *= PML_Qd(xPML);
        Qx[xPML][y][z][xSE] += PML_Qi(xPML) * (Vx(x+1,y,z) - Vx(x,y,z));
    }
    __syncthreads();
    if (ySE>=0) {
        Qy[x][yPML][z][ySE] += PML_Qi(yPML) * (Vy(x,y+1,z) - Vy(x,y,z));
        P(x,y,z)         += PML_Pd(yPML) * Qy[x][yPML][z][ySE];
        Qy[x][yPML][z][ySE] *= PML_Qd(yPML);
        Qy[x][yPML][z][ySE] += PML_Qi(yPML) * (Vy(x,y+1,z) - Vy(x,y,z));
    }
    __syncthreads();
    if (zSE>=0) {
        Qz[x][y][zPML][zSE] += PML_Qi(zPML) * (Vz(x,y,z+1) - Vz(x,y,z));
        P(x,y,z)         += PML_Pd(zPML) * Qz[x][y][zPML][zSE];
        Qz[x][y][zPML][zSE] *= PML_Qd(zPML);
        Qz[x][y][zPML][zSE] += PML_Qi(zPML) * (Vz(x,y,z+1) - Vz(x,y,z));
    }
    __syncthreads();
    // (A) Update P from V.diff
    P(x,y,z) += c * (Vx(x,y,z) - Vx(x+1,y,z) +
                     Vy(x,y,z) - Vy(x,y+1,z) +
                     Vz(x,y,z) - Vz(x,y,z+1) );
}


template <typename X, typename I>
__global__ void
run_rows_V(
    X stepRate,
    X stepAlpha,
    CuArray<X, 4> G,
    const CuArray<I, 3> wallMask,
    const CuArray<X, 2> PMLCoefs)
{
    I nPML = PMLCoefs.size(1);
    I nX = G.size(1) - 1;
    I nY = G.size(2) - 1;
    I nZ = G.size(3) - 1;
    
    X c = stepRate;
    X a = stepAlpha;
    X c1 = c*(1+2*a);
    X ca = c*a;
    
    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int z = threadIdx.x;

    if (x >= nX || y >= nY || z >= nZ) {
        return;
    }

    // Do passes (D),(E),(F)
    int xPMLs = 0;
    int xPMLe = 1;
    int xSE;
    int xPML;
    if (x < nPML) {
        xSE = xPMLs;
        xPML = x;
    } else if (x < nX - nPML) {
        xSE = -1;
        xPML = 1;
    } else if (x < nX - 1) {
        xSE = xPMLe;
        xPML = nX - 2 - x;
    } else {
        xSE = -1;
        xPML = 0;
    }

    // update_Vx()
    // attenuate V
    if (xSE>=0) Vx(x+1,y,z) *= PML_Vd(xPML);
    // (D) Update V from P.diff
    Vx(x+1,y,z) += c1 * ( P(x,y,z) - P(x+1,y,z) );
    if (wM(x-0,y,z,0)) Vx(x+1,y,z) += ca * ( P(x  ,y,z) - P(x-1,y,z) );
    if (wM(x+2,y,z,0)) Vx(x+1,y,z) += ca * ( P(x+2,y,z) - P(x+1,y,z) );
    // (E) Update V from P.diff (PML)
    if (xSE>=0) Vx(x+1,y,z) += PML_Vi(xPML) * -( P(x,y,z) - P(x+1,y,z) );

    __syncthreads();

    // update_Vy()
    int ySE=-1, yPML=nPML;
    if (y<nPML) { ySE = 0; yPML = y; }
    else if (y>nY-2-nPML) { ySE = 1; yPML = nY-2-y; }
    if (y==nY-1) ySE=-1;

    // attenuate V
    if (ySE>=0) Vy(x,y+1,z) *= PML_Vd(yPML);
    // (D) Update V from P.diff
    Vy(x,y+1,z) += c1 * ( P(x,y,z) - P(x,y+1,z) );
    if (wM(x,y-0,z,1)) Vy(x,y+1,z) += ca * ( P(x,y  ,z) - P(x,y-1,z) );
    if (wM(x,y+2,z,1)) Vy(x,y+1,z) += ca * ( P(x,y+2,z) - P(x,y+1,z) );
    // (E) Update V from P.diff (PML)
    if (ySE>=0) Vy(x,y+1,z) += PML_Vi(yPML) * -( P(x,y,z) - P(x,y+1,z) );

    __syncthreads();

    // update_Vz()
    int zSE=-1, zPML=nPML;
    if (z<nPML) { zSE = 0; zPML = z; }
    else if (z>nZ-2-nPML) { zSE = 1; zPML = nZ-2-z; }
    if (z==nZ-1) zSE=-1;
    
    // attenuate V
    if (zSE>=0) Vz(x,y,z+1) *= PML_Vd(zPML);
    // (D) Update V from P.diff
    Vz(x,y,z+1) += c1 * ( P(x,y,z) - P(x,y,z+1) );
    if (wM(x,y,z-0,2)) Vz(x,y,z+1) += ca * ( P(x,y,z  ) - P(x,y,z-1) );
    if (wM(x,y,z+2,2)) Vz(x,y,z+1) += ca * ( P(x,y,z+2) - P(x,y,z+1) );
    // (E) Update V from P.diff (PML)
    if (zSE>=0) Vz(x,y,z+1) += PML_Vi(zPML) * -( P(x,y,z) - P(x,y,z+1) );
}


std::vector<torch::Tensor>
pml_step_cuda(
    cudaStream_t  stream,
    torch::Tensor G,
    float stepRate,
    float stepAlpha,
    torch::Tensor wallMask,
    torch::Tensor Qx2,     
    torch::Tensor Qy2,     
    torch::Tensor Qz2,
    torch::Tensor PMLCoefs)
{
    const auto nPML = PMLCoefs.size(1);
    const auto nX = G.size(1) - 1;
    const auto nY = G.size(2) - 1;
    const auto nZ = G.size(3) - 1;

    auto dtype   = G.scalar_type();
    auto options = torch::TensorOptions().dtype(dtype).device(G.device());

    const int nthreads = nZ;
    const dim3 nblocks(nX, nY);

#define X cuscalar_t
#define I cuint_t

    CUARRAY_KERNEL(
        run_rows_P,
        dtype,
        stream,
        nblocks,
        nthreads,
        stepRate,
        stepAlpha,
        cuarr<X, 4>(G),
        cuarr<X, 4>(Qx2),
        cuarr<X, 4>(Qy2),
        cuarr<X, 4>(Qz2),
        cuarr<X, 2>(PMLCoefs)
    )
    CUARRAY_KERNEL(
        run_rows_V,
        dtype,
        stream,
        nblocks,
        nthreads,
        stepRate,
        stepAlpha,
        cuarr<X, 4>(G),
        cuarr<I, 3>(wallMask),
        cuarr<X, 2>(PMLCoefs)
    )

#undef X
#undef I
    return {G};
}
