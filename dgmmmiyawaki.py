# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:25:26 2021

@author: rolly maulana awangga
"""
# In[]: jika terjadi error pada saat running cnn gunakan ini
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# In[]:
import os    
os.environ['THEANO_FLAGS'] = "device=gpu"  
import numpy as np
import matplotlib.pyplot as plt
#from scipy.io import savemat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from numpy import random
from tensorflow.keras import optimizers
#import matlab.engine
#eng=matlab.engine.start_matlab()
from tensorflow.keras import metrics

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from dgmm import loadtrainandlabel,loadtestandlabel
from lib.bdtb import simpanMSE, simpanMSEMiyawaki, plotDGMM,ubahkelistofchunks,simpanScore



matlist=[]
matlist.append('./de_s1_V1_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
#matlist.append('../de_s1_V2_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
#matlist.append('../de_s1_V1V2_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
#matlist.append('../de_s1_V3VP_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')
#matlist.append('../de_s1_AllArea_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_leave0_1x1_preprocessed.mat')

# In[]: train and predict rolly
matfile=matlist[0]
train_data,label=loadtrainandlabel(matfile)
testdt,testlb=loadtestandlabel(matfile)
predm,labelm,msem=simpanMSEMiyawaki()
# In[]: Load dataset, dengan train dan test bentuk menggunakan testdt dan testlb saja


x=testlb.astype('float32')
y=testdt.astype('float32')
z=predm.astype('float32')

X_train, X_test, Y_train, Y_test, Miyawaki_1, Miyawaki_2 = train_test_split( x, y, z,test_size=20, random_state=7)
#tambah data validasi karena indikasi kecurangan
#X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=7)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=7)
# Pembagian Dataset tanpa random
# fmri=testdt.astype('float32')
# pict=testlb.astype('float32')

# Y_train = fmri[:100]
# Y_test = fmri[-20:]

# X_train = pict[:100]
# X_test = pict[-20:]

# # In[]: Load dataset, yg beda test itu bentuk train acak. 
# Y_train = train_data.astype('float32')
# Y_test = testdt.astype('float32')

# X_train = label#90 gambar dalam baris isi per baris 784 kolom
# X_test = testlb#10 gambar dalam baris isi 784 kolom
# X_train = X_train.astype('float32') / 255.
# X_test = X_test.astype('float32') / 255.


# # In[]: lihat isinya, ketika dijalankan hasilnya jelek
# stim0=np.reshape(X_test[0],(10,10)).T
# stim1=np.reshape(X_test[1],(10,10)).T
# stim2=np.reshape(X_test[2],(10,10)).T
# stim3=np.reshape(X_test[3],(10,10)).T

# stimtrain0=np.reshape(X_train[0],(10,10)).T
# stimtrain1=np.reshape(X_train[1],(10,10)).T
# stimtrain2=np.reshape(X_train[2],(10,10)).T
# stimtrain3=np.reshape(X_train[3],(10,10)).T


# In[]: X adalah gambar stimulus,ukuran pixel 28x28 = 784 di flatten sebelumnya dalam satu baris, 28 row x 28 column dengan channel 1(samaa kaya miyawaki)
resolution = 10#sebelumnya 28
#channel di depan
#X_train = X_train.reshape([X_train.shape[0], 1, resolution, resolution])
#X_test = X_test.reshape([X_test.shape[0], 1, resolution, resolution])
#channel di belakang(edit rolly)
X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
# In[]: Normlization sinyal fMRI
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
Y_train = min_max_scaler.fit_transform(Y_train)     
Y_test = min_max_scaler.transform(Y_test)

print ('X_train.shape : ')
print (X_train.shape)
print ('Y_train.shape')
print (Y_train.shape)
print ('X_test.shape')
print (X_test.shape)
print ('Y_test.shape')
print (Y_test.shape)
numTrn=X_train.shape[0]
numTest=X_test.shape[0]

# In[]: Set the model parameters and hyper-parameters
maxiter = 500
nb_epoch = 1
batch_size = 10
#resolution = 28
D1 = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
D2 = Y_train.shape[1]
K = 6 #panjang fitur untuk Z (latent space)
C = 5 # membentuk diagonal array nilai 1 ditengahnya ukuran CxC(mungkin matrix identitas)
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
numTrn=X_train.shape[0]
#numTest=X_test.shape[0]
numTest=X_validation.shape[0]
# input image dimensions
img_rows, img_cols, img_chns = resolution, resolution, 1

# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

if backend.image_data_format() == 'channels_first': # atau 'channels_last'
    original_img_size = (img_chns, img_rows, img_cols)#1,28, 28
else:
    original_img_size = (img_rows, img_cols, img_chns)#28, 28, 1


# In[]: earsitektur encoder untuk menentukan Z/latent space
#input kontatenasi dari stimulus dan sinyal fMRI, output berupa Z/latent space sebanyak K
X = Input(shape=original_img_size)
Y = Input(shape=(D2,))
Y_mu = Input(shape=(D2,))
Y_lsgms = Input(shape=(D2,))

conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu', name='en_conv_1')(X)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2), name='en_conv_2')(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1, name='en_conv_3')(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1, name='en_conv_4')(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu', name='en_dense_5')(flat)

Z_mu = Dense(K, name='en_mu')(hidden)
Z_lsgms = Dense(K, name='en_var')(hidden)


def sampling(args):
    
    Z_mu, Z_lsgms = args
    epsilon = backend.random_normal(shape=(backend.shape(Z_mu)[0], K), mean=0., stddev=1.0)
    
    return Z_mu + backend.exp(Z_lsgms) * epsilon

Z = Lambda(sampling, output_shape=(K,))([Z_mu, Z_lsgms])
# In[]: Memperlihatkan jumlah fitur output Z sebelum dan sesudah layer lambda
print (Z_mu.shape)
print (Z_lsgms.shape)
print (Z)
# In[]: arsitektur decoder untuk merekonstruksi citra(X_mu,X_lsmgs) sebagai outputan dengna inputan Z 
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 5 * 5, activation='relu')

if backend.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 5, 5)
else:
    output_shape = (batch_size, 5, 5, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if backend.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 29, 29)
else:
    output_shape = (batch_size, 29, 29, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
#yang membedakan X_mu dan X_lsgms adalah fungsi aktifasinya
decoder_mean_squash_mu = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

decoder_mean_squash_lsgms= Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='tanh')
#merangkai arsitekturnya disini setelah di atas mendefinisikan setiap layer
hid_decoded = decoder_hid(Z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)

X_mu = decoder_mean_squash_mu (x_decoded_relu)
X_lsgms = decoder_mean_squash_lsgms (x_decoded_relu)

# In[]: bangun loss function dan 4 model arsitektur, DGMM, encoder, imagepredict dan imagereconstruct
#Membangun loss function
logc = np.log(2 * np.pi).astype(np.float32)
def X_normal_logpdf(x, mu, lsgms):
    lsgms = backend.flatten(lsgms)   
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((x - mu)**2 / backend.exp(lsgms)), axis=-1)

def Y_normal_logpdf(y, mu, lsgms):  
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((y - mu)**2 / backend.exp(lsgms)), axis=-1)
   
def obj(X, X_mu):#loss function antara stimulus X dengan citra rekonstruksi X_mu
    X = backend.flatten(X)
    X_mu = backend.flatten(X_mu)
    
    Lp = 0.5 * backend.mean( 1 + Z_lsgms - backend.square(Z_mu) - backend.exp(Z_lsgms), axis=-1)     
    
    Lx =  - metrics.binary_crossentropy(X, X_mu) # Pixels have a Bernoulli distribution  
               
    Ly =  Y_normal_logpdf(Y, Y_mu, Y_lsgms) # Voxels have a Gaussian distribution
        
    lower_bound = backend.mean(Lp + 10000 * Lx + Ly)
    
    cost = - lower_bound
              
    return  cost 
#bangun model DGMM basis autoencoder dengan inputan extra fMRI
DGMM = Model(inputs=[X, Y, Y_mu, Y_lsgms], outputs=X_mu)
opt_method = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
DGMM.compile(optimizer = opt_method, loss = obj)
print("objective function definisikan")
DGMM.summary()

# bangun model encoder dari inputs stimulus X menjadi Z yang merupakan latent space
encoder = Model(inputs=X, outputs=[Z_mu,Z_lsgms])

# Bangun model autoencoder dari input stimulus X menjadi citra rekonstruksi X_mu dan X_lsmgs
imagepredict = Model(inputs=X, outputs=[X_mu,X_lsgms])

# membangun model decoder rekonstruksi untuk testing dari data test, inputan Z(dimensi K) output gambar
Z_predict = Input(shape=(K,))
_hid_decoded = decoder_hid(Z_predict)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
X_mu_predict = decoder_mean_squash_mu(_x_decoded_relu)
X_lsgms_predict = decoder_mean_squash_mu(_x_decoded_relu)
imagereconstruct = Model(inputs=Z_predict, outputs=X_mu_predict)

# In[]: inisiasi nilai parameter inputan berupa nilai random dahulu, dan dari settingan param sebelumnya
Z_mu = np.mat(random.random(size=(numTrn,K))).astype(np.float32)
B_mu = np.mat(random.random(size=(K,D2))).astype(np.float32)
R_mu = np.mat(random.random(size=(numTrn,C))).astype(np.float32)
sigma_r = np.mat(np.eye((C))).astype(np.float32)
H_mu = np.mat(random.random(size=(C,D2))).astype(np.float32)
sigma_h = np.mat(np.eye((C))).astype(np.float32)

tau_mu = tau_alpha / tau_beta
eta_mu = eta_alpha / eta_beta
gamma_mu = gamma_alpha / gamma_beta
#menentukan nilai Y_mu dan Y_lsgms dari nilai random inisiasi
Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu).astype(np.float32)#dapat dari nilai random
Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2))).astype(np.float32)


from lib import siamese,calculate
#S=np.mat(siamese.S(k, t, Y_train, Y_validation))
S=np.mat(calculate.S(k, t, Y_train, Y_validation))

#savemat('data.mat', {'Y_train':Y_train,'Y_test':Y_test})
#S=np.mat(eng.calculateS(float(k), float(t))).astype(np.float32)
# In[]: Y fMRI input, Y_mu didapat dari nilai random, Y_lsgms nilai log
print (X_train.shape)
print (Y_train.shape)
print (Y_mu.shape)
print (Y_lsgms.shape)

# In[]: Loop training Y_mu dan Y_lsgms berubah terus setiap iterasi, optimasi di Z
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
    # update B dari hasil Z_mu dan Z_lsgms
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
    # calculate Y_mu dari random, Y_lsgms dari Y_Train untuk input loop selanjutnya   
    Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu) 
    Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))   

# In[]: reconstruct X (image) from Y (fmri)
X_reconstructed_mu = np.zeros((numTest, img_chns, img_rows, img_cols))
HHT = H_mu * H_mu.T + D2 * sigma_h
Temp = gamma_mu * np.mat(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.mat(np.eye(C)) + gamma_mu * HHT).I * H_mu)
for i in range(numTest):
    s=S[:,i]
    z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.mat(np.eye(K)) ).I
    z_mu_test = (z_sigma_test * (B_mu * Temp * (np.mat(Y_test)[i,:]).T + rho * np.mat(Z_mu).T * s )).T
    temp_mu = np.zeros((1,img_chns, img_rows, img_cols))#1,1,28,28
    epsilon_std = 1
    for l in range(L):#denoising monte carlo
        epsilon=np.random.normal(0,epsilon_std,1)
        z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test))*epsilon
        x_reconstructed_mu = imagereconstruct.predict(z_test, batch_size=1)#1,28,28,1
        #edit rolly move axis
        x_reconstructed_mu=np.moveaxis(x_reconstructed_mu,-1,1)
        temp_mu = temp_mu + x_reconstructed_mu # ati2 nih disini main tambahin aja
    x_reconstructed_mu = temp_mu / L
    X_reconstructed_mu[i,:,:,:] = x_reconstructed_mu

# In[]:# visualization the reconstructed images
n = 20
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


# In[]: Hitung FID dan IS dan MSE
# Existing code
stim = X_test[:, :, :, 0].reshape(20, 100)
rec = X_reconstructed_mu[:, 0, :, :].reshape(20, 100)

from lib.fidis import save_array_as_image
# Save stim array as images
for i in range(len(stim)):
    save_array_as_image(stim[i].reshape(10, 10), f'stim/image_{i}.png')

# Save rec array as images
for i in range(len(rec)):
    save_array_as_image(rec[i].reshape(10, 10), f'rec/image_{i}.png')

# Calculate FID - Frechet Inception Distance (FID)
#fid_value = calculate_fid("stim","rec")
#print('FID:', fid_value)

# Calculate IS
#is_mean, is_std = calculate_is(generated_images)
#print('Inception Score - Mean:', is_mean)
#print('Inception Score - Std:', is_std)

# Continue with the rest of your existing code
scoreresults = simpanScore(stim, rec, matfile, 'VAE')
scoreresults_miyawaki = simpanScore(stim, Miyawaki_2, matfile, 'Miyawaki')

mse = simpanMSE(stim, rec, matfile, 'VAE')
msem = simpanMSE(stim, Miyawaki_2, matfile, 'miyawaki')

chunk = 10
lmse, lmsem, lpred, lpredm, llabel = ubahkelistofchunks(mse, msem, rec, Miyawaki_2, stim, chunk)

n = 1
for label, pred, predm, mse, msem in zip(llabel, lpred, lpredm, lmse, lmsem):
    plotDGMM(label, pred, predm, mse, msem, matfile, n, 'VAE')
    n = n + 1


# In[]:
np.savetxt('skorvae.csv',scoreresults,delimiter=',')
np.savetxt('skormiyawaki.csv',scoreresults_miyawaki,delimiter=',')