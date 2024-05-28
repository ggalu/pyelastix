# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7 2018

@author: gcg

Purpose: convert an elastix/transformix full spatial Jacobian field (containing the
deformation gradient F) to Green-Lagrange strains.
"""
import os, sys
import numpy as np

import jax
from jax import jit
import jax.numpy as jnp
import tifffile
import progressbar
import os

class StrainComputer:
    def __init__(self, filename, margin=50):

        self.path = os.path.dirname(os.path.abspath(filename))
        self.eye = np.identity(3)
        self.F = tifffile.imread(filename)
        self.nz, self.ny, self.nx, _ = self.F.shape
        print("shape of F:", self.F.shape)
        self.margin = margin
        print("zero margin (top and bottom):", self.margin)


        self.compute_strains()
        self.save_E() # save stretch tensor diagonals

    def compute_strains(self):
        """ sequentially read z-slices and compute strains.
        """

        @jit
        def compute_stretch_worker(M):
            """ compute Green Lagrange strain and return only the diagonals
            """
            E = 0.5 * (jnp.matmul(M.transpose(),M) - self.eye)
            return jnp.diagonal(E)

        def compute_stretches(array):
            result = jnp.zeros((self.ny*self.nx,3)) # double precision
            result = jax.vmap(compute_stretch_worker)(array)
            return result.astype(np.float32)

        #self.eigvals = np.zeros((nz,ny,nx,3), dtype=np.float32)
        self.E = np.zeros((self.nz,self.ny,self.nx,3), dtype=np.float32)

        for iz in  progressbar.progressbar(range(self.margin, self.nz-self.margin)):
            thisF = self.F[iz] # process one z-slice of deformation gradient 
            flatF = np.reshape(thisF,(self.ny*self.nx,3,3)) # reshape this to have a linear array of 3x3 tensors
            self.E[iz,:,:,:] = compute_stretches(flatF).reshape(self.ny,self.nx,3) # # compute stretch tensor diagonals and put back into original shape

    def save_E(self):
        outfile = os.path.join(self.path, "Exx.tif")            
        tifffile.imwrite(outfile, self.E[:,:,:,0])

        outfile = os.path.join(self.path, "Eyy.tif")            
        tifffile.imwrite(outfile, self.E[:,:,:,2])

        outfile = os.path.join(self.path, "Ezz.tif")            
        tifffile.imwrite(outfile, self.E[:,:,:,2])
    

if __name__ == "__main__":
    strainComputer = StrainComputer("nonrigid/fullSpatialJacobian.tif")