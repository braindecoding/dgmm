# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:47:17 2024

@author: DTI
"""
from pytorch_fid import fid_score

def calculate_fid(real_images, generated_images, batch_size=20):
    fid = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=min(batch_size, len(real_images)), device='cpu', dims=2048)
    return fid

