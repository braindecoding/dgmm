#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:50:17 2017
https://github.com/ChangdeDu/DGMM
@author: duchangde 
"""

import os    
os.environ['THEANO_FLAGS'] = "device=gpu"  
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from numpy import random
from tensorflow.keras import optimizers
import matlab.engine
eng=matlab.engine.start_matlab()
from tensorflow.keras import metrics

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from lib import prepro,ars,obj,init,calculate

# In[]: Load dataset X stimulus Y fMRI
resolution = 28
X_train,X_test,Y_train,Y_test=prepro.getXY('digit69_28x28.mat',resolution)

# In[]: Set the model parameters and hyper-parameters
maxiter = 200
nb_epoch = 1
batch_size = 10
D1 = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
D2 = Y_train.shape[1]
K = 6 # dimensi latent space
C = 5
intermediate_dim = 128

#hyper-parameters
tau_alpha = 1
tau_beta = 1
eta_alpha = 1
eta_beta = 1
gamma_alpha = 1
gamma_beta = 1

Beta = 1 # Beta-VAE for Learning Disentangled Representations
rho=0.1  # posterior regularization parameter

k=10     # k-nearest neighbors
t = 10.0 # kernel parameter in similarity measure

L = 100   # Monte-Carlo sampling

np.random.seed(1000)
numTrn=X_train.shape[0]#ada 90 data training
numTest=X_test.shape[0]#ada 10 data testing

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1

# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

if backend.image_data_format() == 'channels_first': # atau 'channels_last'
    original_img_size = (img_chns, img_rows, img_cols)#1,28, 28
else:
    original_img_size = (img_rows, img_cols, img_chns)#28, 28, 1

# In[]:     
#savemat('data.mat', {'Y_train':Y_train,'Y_test':Y_test})

#S=np.mat(eng.calculateS(float(k), float(t))).astype(np.float32)
#s=S[:,0]

S=np.mat(calculate.S(k, t, Y_train, Y_test))
#s=S[:,0]



##### testing only rolly
#for i in range(numTest):
    #print("for i in range(numTest):")
    #print(i)
    #s=S[:,i]
    #print(s)

# In[]: Building the architechture
#input arsitektur dimensi stimulus
X = Input(shape=original_img_size)
#input arsitektur dimensi fmri
Y = Input(shape=(D2,))
Y_mu = Input(shape=(D2,))
Y_lsgms = Input(shape=(D2,))

Z,Z_lsgms,Z_mu = ars.encoder(X, D2, img_chns, filters, num_conv, intermediate_dim, K)

# In[]: we instantiate these layers separately so as to reuse them later
decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms=ars.decoderars(intermediate_dim, filters, batch_size, num_conv, img_chns)
# arsitekturnya dipake di bawahnya ketika compile model, jika ingin membuat layer baru lagi :
#Namun, perlu diingat bahwa dengan pendekatan ini, Anda tidak akan dapat menggunakan kembali layer yang sama, karena setiap kali Anda memanggil fungsi decoder, layer baru akan dibuat. Jika Anda ingin menggunakan kembali layer yang sama, Anda harus tetap membuatnya di luar fungsi dan mengirimkannya sebagai argumen, seperti yang Anda lakukan sebelumnya.
#X_mu,X_lsgms=ars.decoder(Z,intermediate_dim, filters, batch_size, num_conv, img_chns)

X_mu,X_lsgms=ars.decoders(Z, decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms)

# In[]:define custom loss objective function   
def custom_loss(X, X_mu):#stimulus asli dan hasil pembangkitan
    X = backend.flatten(X)
    X_mu = backend.flatten(X_mu) 
    Lp = 0.5 * backend.mean( 1 + Z_lsgms - backend.square(Z_mu) - backend.exp(Z_lsgms), axis=-1)     
    Lx =  - metrics.binary_crossentropy(X, X_mu) # Pixels have a Bernoulli distribution  
    Ly =  obj.Y_normal_logpdf(Y, Y_mu, Y_lsgms,backend) # Voxels have a Gaussian distribution
    lower_bound = backend.mean(Lp + 10000 * Lx + Ly)
    cost = - lower_bound
    return  cost 


#Jika Y, Y_mu, dan Y_lsgms hanya digunakan dalam perhitungan kerugian dan tidak berkontribusi langsung pada perhitungan output dari model, mereka mungkin berfungsi sebagai "label" tambahan yang membantu model belajar representasi yang lebih baik dari data dengan memberikan lebih banyak informasi tentang bagaimana kerugian harus dihitung.
DGMM = Model(inputs=[X, Y, Y_mu, Y_lsgms], outputs=X_mu)

try:
    opt_method = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
except:
    opt_method = optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    

DGMM.compile(optimizer = opt_method, loss = custom_loss)
DGMM.summary()

# build a model to project inputs on the latent space
encoder = Model(inputs=X, outputs=[Z_mu,Z_lsgms])
# build a model to project inputs on the output space
imagepredict = Model(inputs=X, outputs=[X_mu,X_lsgms])

# build a digit generator that can sample from the learned distribution
Z_predict = Input(shape=(K,))

X_mu_predict,X_lsgms_predict=ars.decoders(Z_predict, decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms)

imagereconstruct = Model(inputs=Z_predict, outputs=X_mu_predict)

# In[]: Initialization

Z_mu,B_mu,R_mu,H_mu=init.random0or1withmatrixsize(numTrn, K, C, D2)
Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu).astype(np.float32)

sigma_r,sigma_h = init.matriksidentitasukuran(C)

tau_mu,eta_mu,gamma_mu=init.alphabagibeta(tau_alpha,tau_beta,eta_alpha,eta_beta,gamma_alpha,gamma_beta)
Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2))).astype(np.float32)

#savemat('data.mat', {'Y_train':Y_train,'Y_test':Y_test})
#S=np.mat(eng.calculateS(float(k), float(t))).astype(np.float32)
#S=calculate.S(k, t, Y_train, Y_test)

# In[]: Loop training
for l in range(maxiter):
    print ('**************************************     iter= ', l)
    # update Z
    DGMM.fit([X_train, Y_train, Y_mu, Y_lsgms], X_train,
            shuffle=True,
            verbose=2,
            epochs=nb_epoch,
            batch_size=batch_size)         
    [Z_mu,Z_lsgms] = encoder.predict(X_train) 
    Z_mu = np.mat(Z_mu) 
    # update B
    temp1 = np.exp(Z_lsgms)
    temp2 = Z_mu.T * Z_mu + np.mat(np.diag(temp1.sum(axis=0)))
    temp3 = tau_mu * np.mat(np.eye(K))
    sigma_b = (gamma_mu * temp2 + temp3).I
    B_mu = sigma_b * gamma_mu * Z_mu.T * (np.mat(Y_train) - R_mu * H_mu)
    # update H
    RTR_mu = R_mu.T * R_mu + numTrn * sigma_r
    sigma_h = (eta_mu * np.mat(np.eye(C)) + gamma_mu * RTR_mu).I
    H_mu = sigma_h * gamma_mu * R_mu.T * (np.mat(Y_train) - Z_mu * B_mu)
    # update R
    HHT_mu = H_mu * H_mu.T + D2 * sigma_h
    sigma_r = (np.mat(np.eye(C)) + gamma_mu * HHT_mu).I
    R_mu = (sigma_r * gamma_mu * H_mu * (np.mat(Y_train) - Z_mu * B_mu).T).T  
    # update tau
    tau_alpha_new = tau_alpha + 0.5 * K * D2
    tau_beta_new = tau_beta + 0.5 * ((np.diag(B_mu.T * B_mu)).sum() + D2 * sigma_b.trace())
    tau_mu = tau_alpha_new / tau_beta_new
    tau_mu = tau_mu[0,0] 
    # update eta
    eta_alpha_new = eta_alpha + 0.5 * C * D2
    eta_beta_new = eta_beta + 0.5 * ((np.diag(H_mu.T * H_mu)).sum() + D2 * sigma_h.trace())
    eta_mu = eta_alpha_new / eta_beta_new
    eta_mu = eta_mu[0,0] 
    # update gamma
    gamma_alpha_new = gamma_alpha + 0.5 * numTrn * D2
    gamma_temp = np.mat(Y_train) - Z_mu * B_mu - R_mu * H_mu
    gamma_temp = np.multiply(gamma_temp, gamma_temp)
    gamma_temp = gamma_temp.sum(axis=0)
    gamma_temp = gamma_temp.sum(axis=1)
    gamma_beta_new = gamma_beta + 0.5 * gamma_temp
    gamma_mu = gamma_alpha_new / gamma_beta_new
    gamma_mu = gamma_mu[0,0] 
    # calculate Y_mu   
    Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu) 
    Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))   

# In[]: save model and parameter results
import pickle
with open('dgmm.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([numTest, img_chns, img_rows, img_cols,H_mu,D2,sigma_h,gamma_mu,C,S,B_mu,rho,K,Y_test,Z_mu,L,resolution,X_test], f)

imagereconstruct.save("dgmmmodel.h5")

# In[]: reconstruct X (image) from Y (fmri)
print("reconstruct X (image) from Y (fmri)")
X_reconstructed_mu = np.zeros((numTest, img_chns, img_rows, img_cols))
HHT = H_mu * H_mu.T + D2 * sigma_h
Temp = gamma_mu * np.mat(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.mat(np.eye(C)) + gamma_mu * HHT).I * H_mu)
for i in range(numTest):
    print("for i in range(numTest):")
    print(i)
    s=S[:,i]
    print(s)
    z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.mat(np.eye(K)) ).I
    z_mu_test = (z_sigma_test * (B_mu * Temp * (np.mat(Y_test)[i,:]).T + rho * np.mat(Z_mu).T * s )).T
    temp_mu = np.zeros((1,img_chns, img_rows, img_cols))#1,1,28,28
    epsilon_std = 1
    for l in range(L):
        epsilon=np.random.normal(0,epsilon_std,1)
        z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test))*epsilon
        x_reconstructed_mu = imagereconstruct.predict(z_test, batch_size=1)#1,28,28,1
        #edit rolly move axis
        x_reconstructed_mu=np.moveaxis(x_reconstructed_mu,-1,1)
        temp_mu = temp_mu + x_reconstructed_mu # ati2 nih disini main tambahin aja
    x_reconstructed_mu = temp_mu / L
    X_reconstructed_mu[i,:,:,:] = x_reconstructed_mu

# In[]:# visualization the reconstructed images, output in var X_reconstructed_mu
n = 10
for j in range(1):
    plt.figure(figsize=(12, 2))    
    for i in range(n):
        # display original images
        ax = plt.subplot(2, n, i +j*n*2 + 1)
        plt.imshow(np.rot90(np.fliplr(X_test[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(2, n, i + n + j*n*2 + 1)
        plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
