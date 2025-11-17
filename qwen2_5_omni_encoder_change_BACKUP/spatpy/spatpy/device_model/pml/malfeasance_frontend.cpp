#include "fem_grid_pml.h"
#include <cstdio>
#include <filesystem>
#include <string>

int main(int argc, char *argv[]) {
    float    stepRate = 0.5;
    float    stepAlpha = 0.07735026918962584;
    void    *G;
    void *wallInds;
    void *inInds;  
    void    *inSig;   
    void  *wallMask;
    void    *Qx2;     
    void    *Qy2;     
    void    *Qz2;
    void    *PMLCoefs;
    void *outInds;
    void  *PML_en; 
    void *pLossInds;
    void *sizes;

    std::filesystem::path p = argc > 1 ? std::string(argv[1]) : ".";
    read_bin((p / "sizes.bin").c_str(), sizeof(uint64_t), &sizes, NULL);
    uint64_t *sz = static_cast<uint64_t *>(sizes);
    uint64_t nX = sz[0];
    uint64_t nY = sz[1];
    uint64_t nZ = sz[2];
    uint64_t nWalls = sz[3];
    uint64_t nIn = sz[4];
    uint64_t nOut = sz[5];
    uint64_t nPML = sz[6];
    uint64_t nSamp = sz[7];
    uint64_t nPLoss = sz[8];
    float *outSig = (float *) malloc(nSamp * nOut * sizeof(float));
    read_bin((p / "inSig.bin").c_str(), sizeof(float), &inSig, NULL);
    read_bin((p / "G.bin").c_str(), sizeof(float), &G, NULL);
    read_bin((p / "wallInds.bin").c_str(), sizeof(uint64_t), &wallInds, &nWalls);
    read_bin((p / "inInds.bin").c_str(), sizeof(uint64_t), &inInds, NULL);
    read_bin((p / "outInds.bin").c_str(), sizeof(uint64_t), &outInds, NULL);
    read_bin((p / "pLossInds.bin").c_str(), sizeof(uint64_t), &pLossInds, &nPLoss);
    read_bin((p / "Qx2.bin").c_str(), sizeof(float), &Qx2, NULL);
    read_bin((p / "Qy2.bin").c_str(), sizeof(float), &Qy2, NULL);
    read_bin((p / "Qz2.bin").c_str(), sizeof(float), &Qz2, NULL);
    read_bin((p / "PMLCoefs.bin").c_str(), sizeof(float), &PMLCoefs, NULL);
    read_bin((p / "wallMask.bin").c_str(), sizeof(uint8_t), &wallMask, NULL);
    read_bin((p / "PML_en.bin").c_str(), sizeof(uint8_t), &PML_en, NULL);
    fem_grid_pml_step(
        outSig,
        static_cast<float *>(G),
        stepRate,
        stepAlpha,
        static_cast<uint64_t *>(wallInds),
        static_cast<uint64_t *>(inInds),
        static_cast<float *>(inSig),
        static_cast<uint8_t *>(wallMask),
        static_cast<float *>(Qx2),     
        static_cast<float *>(Qy2),     
        static_cast<float *>(Qz2),
        static_cast<float *>(PMLCoefs),
        static_cast<uint64_t *>(outInds),
        static_cast<uint8_t *>(PML_en), 
        static_cast<uint64_t *>(pLossInds),
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

    return 0;
}
