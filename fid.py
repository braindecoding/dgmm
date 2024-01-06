# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:47:17 2024

@author: DTI
"""

from pytorch_fid import fid_score
import sys

def calculate_fid(real_images, generated_images, batch_size=20):
    fid = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=min(batch_size, len(real_images)), device='cpu', dims=2048)
    return fid


if __name__ == '__main__':
    K=int(sys.argv[1])
    intermediate_dim=int(sys.argv[2])
    batch_size=int(sys.argv[3])
    maxiter=int(sys.argv[4])
    experimentname=str(K)+","+str(intermediate_dim)+"," + str(batch_size)+"," + str(maxiter)+","
    # Calculate FID - Frechet Inception Distance (FID)
    fid_value = calculate_fid("stim","rec")
    print('FID:', fid_value)
    with open("FID_Results.csv", "a") as myfile:
        myfile.write(experimentname+","+str(fid_value))