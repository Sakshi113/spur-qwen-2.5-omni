import torch
from torch.utils.cpp_extension import load

import os
import sys
import numpy as np

# adds ninja to the path when using conda
os.environ['PATH'] += ':' + os.path.join(sys.exec_prefix, 'bin')

HERE = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))

if torch.cuda.is_available():
    try:
        print('\x1b[32m\x1b[2m')
        pml_cuda = load(
            name='pml_cuda',
            extra_include_paths=[HERE, BASE],
            sources=[
                os.path.join(HERE, 'pml_cuda.cpp'),
                os.path.join(HERE, 'pml_cuda_kernel.cu')
            ],
            verbose=True
        )
    finally:
        print('\x1b[0m\x1b[0m')
else:
    class pml_cuda:
        def pml_step(self):
            print('Unable to load pml_cuda extension')
            raise NotImplementedError


class FEMGridPMLCUDA:
    def __init__(
        self,
        order,
        g,
        step_rate,
        step_alpha,
        wall_inds,
        in_inds,
        wall_mask,
        Qx2,
        Qy2,
        Qz2,
        pml_coefs,
        out_inds,
        pml_en,
        p_loss_inds,
        device=None
    ):
        if device is None:
            device = torch.device("cuda")
        self.step_rate = step_rate
        self.step_alpha = step_alpha
        self.g = torch.from_numpy(g).to(device=device)
        self.wall_mask = torch.from_numpy(wall_mask).to(device=device, dtype=torch.int16)
        self.Qx2 = torch.from_numpy(Qx2).to(device=device)
        self.Qy2 = torch.from_numpy(Qy2).to(device=device)
        self.Qz2 = torch.from_numpy(Qz2).to(device=device)
        self.pml_coefs = torch.from_numpy(pml_coefs).to(device=device)
        self.wall_inds = torch.from_numpy((wall_inds - 1).astype(np.int64)).to(device=device)
        self.p_loss_inds = (p_loss_inds - 1).astype(np.uint64)
        self.out_inds = (out_inds - 1).astype(np.uint64)
        self.in_inds = (in_inds - 1).astype(np.uint64)
        self.nin = in_inds.shape[0]
        self.nout = out_inds.shape[0]

    def step(self, in_sig):
        nsamp = in_sig.shape[0]
        out_sig = np.zeros((nsamp, self.nout))
        g = torch.flatten(self.g)
        for sample_num in range(nsamp):
            # Apply pressure-loss, to emulate abosrbing wall
            for p in self.p_loss_inds:
                g[p] *= 1 - self.step_rate

            # Get Output samples
            for (i, ind) in enumerate(self.out_inds):
                out_sig[sample_num, i] = g[ind]

            # Add input samples
            for (i, ind) in enumerate(self.in_inds):
                g[ind] += in_sig[sample_num, i]

            pml_cuda.pml_step(self.g, self.step_rate, self.step_alpha, self.wall_mask, self.Qx2, self.Qy2, self.Qz2, self.pml_coefs)
            # Force walls to zero
            g.index_fill_(0, self.wall_inds, 0.0)

        return out_sig
