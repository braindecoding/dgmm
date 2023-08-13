# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:44:06 2023

@author: rolly
"""
import numpy as np

def updateZ(DGMM,X_train, Y_train, Y_mu, Y_lsgms,nb_epoch,batch_size,encoder):
    DGMM.fit([X_train, Y_train, Y_mu, Y_lsgms], X_train,
            shuffle=True,
            verbose=2,
            epochs=nb_epoch,
            batch_size=batch_size)         
    [Z_mu,Z_lsgms] = encoder.predict(X_train) 
    Z_mu = np.mat(Z_mu) 
    return Z_mu,Z_lsgms

def updateB(Z_lsgms,Z_mu,K,tau_mu,gamma_mu,Y_train,R_mu,H_mu):
    temp1 = np.exp(Z_lsgms)
    temp2 = Z_mu.T * Z_mu + np.mat(np.diag(temp1.sum(axis=0)))
    temp3 = tau_mu * np.mat(np.eye(K))
    sigma_b = (gamma_mu * temp2 + temp3).I
    B_mu = sigma_b * gamma_mu * Z_mu.T * (np.mat(Y_train) - R_mu * H_mu)
    return B_mu,sigma_b

def updateH(R_mu,numTrn,sigma_r,eta_mu,C,gamma_mu,Y_train,Z_mu,B_mu):
    RTR_mu = R_mu.T * R_mu + numTrn * sigma_r
    sigma_h = (eta_mu * np.mat(np.eye(C)) + gamma_mu * RTR_mu).I
    H_mu = sigma_h * gamma_mu * R_mu.T * (np.mat(Y_train) - Z_mu * B_mu)
    return H_mu,sigma_h

def updateR(H_mu,D2,sigma_h,C,gamma_mu,Y_train,Z_mu,B_mu):
    HHT_mu = H_mu * H_mu.T + D2 * sigma_h
    sigma_r = (np.mat(np.eye(C)) + gamma_mu * HHT_mu).I
    R_mu = (sigma_r * gamma_mu * H_mu * (np.mat(Y_train) - Z_mu * B_mu).T).T  
    return R_mu,sigma_r

def updateTau(tau_alpha,K,D2,tau_beta,B_mu,sigma_b):
    tau_alpha_new = tau_alpha + 0.5 * K * D2
    tau_beta_new = tau_beta + 0.5 * ((np.diag(B_mu.T * B_mu)).sum() + D2 * sigma_b.trace())
    tau_mu = tau_alpha_new / tau_beta_new
    tau_mu = tau_mu[0,0] 
    return tau_mu

def updateEta(eta_alpha,C,D2,eta_beta,H_mu,sigma_h):
    eta_alpha_new = eta_alpha + 0.5 * C * D2
    eta_beta_new = eta_beta + 0.5 * ((np.diag(H_mu.T * H_mu)).sum() + D2 * sigma_h.trace())
    eta_mu = eta_alpha_new / eta_beta_new
    eta_mu = eta_mu[0,0] 
    return eta_mu

def updateGamma(gamma_alpha,numTrn,D2,Y_train,Z_mu,B_mu,R_mu,H_mu,gamma_beta):
    gamma_alpha_new = gamma_alpha + 0.5 * numTrn * D2
    gamma_temp = np.mat(Y_train) - Z_mu * B_mu - R_mu * H_mu
    gamma_temp = np.multiply(gamma_temp, gamma_temp)
    gamma_temp = gamma_temp.sum(axis=0)
    gamma_temp = gamma_temp.sum(axis=1)
    gamma_beta_new = gamma_beta + 0.5 * gamma_temp
    gamma_mu = gamma_alpha_new / gamma_beta_new
    gamma_mu = gamma_mu[0,0] 
    return gamma_mu