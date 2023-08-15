import numpy as np
from numpy import random


#Z_mu, B_mu, R_mu, dan H_mu diinisialisasi sebagai matriks dengan elemen acak yang dihasilkan oleh fungsi random.random dari library numpy. Fungsi ini menghasilkan angka acak antara 0 dan 1. Ukuran matriks ini ditentukan oleh variabel numTrn, K, C, dan D2.
def randombetween0and1withmatrixsize(numTrn,K,C,D2):
    Z_mu = np.mat(random.random(size=(numTrn,K))).astype(np.float32)
    B_mu = np.mat(random.random(size=(K,D2))).astype(np.float32)
    R_mu = np.mat(random.random(size=(numTrn,C))).astype(np.float32)
    H_mu = np.mat(random.random(size=(C,D2))).astype(np.float32)
    return Z_mu,B_mu,R_mu,H_mu

def matriksidentitasukuran(C):
    sigma_r = np.mat(np.eye((C))).astype(np.float32)
    sigma_h = np.mat(np.eye((C))).astype(np.float32)
    return sigma_r,sigma_h

def alphabagibeta(tau_alpha,tau_beta,eta_alpha,eta_beta,gamma_alpha,gamma_beta):
    tau_mu = tau_alpha / tau_beta
    eta_mu = eta_alpha / eta_beta
    gamma_mu = gamma_alpha / gamma_beta
    return tau_mu,eta_mu,gamma_mu