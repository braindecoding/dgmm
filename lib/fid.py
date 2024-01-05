# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:47:17 2024

@author: DTI
"""
import torch
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pytorch_fid import fid_score

def calculate_fid(real_images, generated_images, batch_size=20):
    fid = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=min(batch_size, len(real_images)), device='cpu', dims=2048)
    return fid

def inception_score(model, images, batch_size, resize=True):
    model.eval()
    preds = []

    loader = DataLoader(images, batch_size=batch_size)

    for batch in tqdm(loader, desc="Calculating Inception Score"):
        #batch = batch.cuda()
        batch = batch.cpu()
        preds.append(model(batch)[0].detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)

    return preds

def calculate_inception_score(images, batch_size=32, resize=True, splits=10):
    # Set up Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()

    # Resize images if necessary
    if resize:
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

    # Calculate Inception Score
    preds = inception_score(inception_model, images, batch_size, resize)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)
