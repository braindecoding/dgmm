# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:15:32 2021

@author: rolly
"""
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import device
from tensorflow.python.client.device_lib import list_local_devices
import scipy.io
from skimage.metrics import structural_similarity
import cv2
import math

def getdatatrainfrommat(matfile):
    mat = scipy.io.loadmat(matfile)
    train_data,label=loadtrainandlabel(mat)
    images=np.delete(label,0,1)
    return train_data,images

def getdatatestfrommat(matfile):
    mat = scipy.io.loadmat(matfile)
    train_data,label=loadtestandlabel(mat)
    images=np.delete(label,0,1)
    return train_data,images


def trainModel(matfile,arch):
    mat = scipy.io.loadmat(matfile)
    train_data,label=loadtrainandlabel(mat)
    for x in range(1,101):#pembentukan model dari pixel 1-100
            labelperpx=getlabel(label,x)#mendapatkan label per pixel
            path=modelfolderpath(matfile)+str(arch)+'\\'+str(x)#melakukan set path model
            createmodel(train_data,labelperpx,path,arch)#membuat dan menyimpan model

def testModel(matfile,arch):
    mat = scipy.io.loadmat(matfile)
    testdt,testlb=loadtestandlabel(mat)
    pixel=1
    path=modelfolderpath(matfile)+str(arch)+'\\'+str(pixel)
    print(path)
    piksel=generatePixel(path,testdt)
    for x in range(2,101):
        path=modelfolderpath(matfile)+str(arch)+'\\'+str(x)
        pikselbr=generatePixel(path,testdt)
        piksel=np.concatenate((piksel,pikselbr),axis=1)
    pxlb=delfirstCol(testlb)
    return pxlb,piksel

def simpanSemuaGambar(pxlb,piksel,matfile):
    n=1
    for stim,recon in zip(pxlb,piksel):
        simpanGambar(stim,recon,getfigpath(matfile,'reconstruct',n))
        n=n+1

def simpanScore(label,pred,matfile,arch):
    allres=np.zeros(shape=(1,4))
    for limagerow,predimagerow in zip(label,pred):
        mlabel=rowtoimagematrix(limagerow)
        mpred=rowtoimagematrix(predimagerow)
        mseresult=msescore(mlabel,mpred)
        ssimresult=ssimscore(mlabel,mpred)
        psnrresult=psnrscore(mlabel,mpred)
        corrresult=corrscore(mlabel,mpred)
        therow=np.array([[mseresult,ssimresult,psnrresult,corrresult]])
        allres=np.concatenate((allres,therow),axis=0)
    fname=msefilename(matfile,arch)
    createfolder(getfoldernamefrompath(fname))
    allres = np.delete(allres, (0), axis=0)
    print(fname)
    np.savetxt(fname,allres,delimiter=',', fmt='%f')
    return allres

def simpanMSE(label,pred,matfile,arch):#matfile digunakan untuk menamai file
    #mse sendiri
    mse = ((label - pred)**2).mean(axis=1)
    fname=msefilename(matfile,arch)
    createfolder(getfoldernamefrompath(fname))
    np.savetxt(fname,mse,delimiter=',')
    return mse
    
def simpanScoreMiyawaki():
    directory='./data/'
    #matfilename='s1_V1_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_figRecon_linComb-no_opt_1x1_maxProbLabel_dimNorm.mat'
    matfilename='s1_V1_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_figRecon_linComb-errFuncImageNonNegCon_1x1_maxProbLabel_dimNorm.mat'
    matfile=directory+matfilename
    mat = scipy.io.loadmat(matfile)
    pred,label=mat['stimFigTestAllPre'],mat['stimFigTestAll']
    allres=np.zeros(shape=(1,4))
    for limagerow,predimagerow in zip(label,pred):
        mlabel=rowtoimagematrix(limagerow)
        mpred=rowtoimagematrix(predimagerow)
        mseresult=msescore(mlabel,mpred)
        ssimresult=ssimscore(mlabel,mpred)
        psnrresult=psnrscore(mlabel,mpred)
        corrresult=corrscore(mlabel,mpred)
        therow=np.array([[mseresult,ssimresult,psnrresult,corrresult]])
        allres=np.concatenate((allres,therow),axis=0)
    allres = np.delete(allres, (0), axis=0)
    np.savetxt('miyawaki.csv',allres,delimiter=',', fmt='%f')
    return pred,label,allres

def simpanMSEMiyawaki():
    directory='./data/'
    #matfilename='s1_V1_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_figRecon_linComb-no_opt_1x1_maxProbLabel_dimNorm.mat'
    matfilename='s1_V1_Ecc1to11_baseByRestPre_smlr_s1071119ROI_resol10_figRecon_linComb-errFuncImageNonNegCon_1x1_maxProbLabel_dimNorm.mat'
    matfile=directory+matfilename
    mat = scipy.io.loadmat(matfile)
    pred,label=mat['stimFigTestAllPre'],mat['stimFigTestAll']
    mse = ((pred - label)**2).mean(axis=1)
    np.savetxt('miyawaki.csv',mse,delimiter=',')
    return pred,label,mse

def testingGPUSupport():
    local_device_protos = list_local_devices()
    print(local_device_protos)

def runOnGPU(model):
    with device('/gpu:0'):
        model.fit()

def loaddatanorest(mat):
    mdata =mat['D']
    mdtype = mdata .dtype 
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}
    label = ndata['label']
    data = ndata['data']
    nl=[]
    nd=[]
    for l,d in zip(label,data):
        if l[1] < 2:
            nl.append(l)
            nd.append(d)
    return nl,nd

def loadtestandlabel(mat):
    nl,nd=loaddatanorest(mat)
    label=nl[440:]
    data=nd[440:]
    return np.asarray(data, dtype=np.float64),np.asarray(label, dtype=np.float64)
    
def loadtrainandlabel(mat):
    nl,nd=loaddatanorest(mat)
    alllabel=nl[:440]
    rdata=nd[:440]
    return np.asarray(rdata, dtype=np.float64),np.asarray(alllabel, dtype=np.float64)

def getlabel(alllabel,x):
    px1=[]
    for i in alllabel:
        px1.append(i[x])        
    label_data=np.asarray(px1, dtype=np.float64)
    return label_data

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
def createmodel(train_data,label_data,filename,arch):
    X = train_data#440row
    y = label_data
    featurelength=len(train_data[0])
    print('feature leength : ')#967
    print(featurelength)
    arcl=arch.split('_')
    print(arcl)
    # define the keras model
    model = Sequential()
    firstarch=int(arcl.pop(0))
    model.add(Dense(firstarch, input_dim=featurelength, activation='sigmoid'))
    for arc in arcl:
        model.add(Dense(int(arc),activation='relu'))
    if firstarch != 1:
        model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=500, batch_size=40)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    model.save(str(filename))
    
def generatePixel(pxpath,data):
    print(pxpath)
    model = load_model(pxpath)
    #return model.predict_classes(data)
    res = model.predict(data)
    #print(res)
    return res

def showFig(az):
    gbr = az.reshape((10,10)).T
    plt.imshow(gbr)

def getfoldernamefrompath(fullpath):
    return fullpath.split('\\')[-2]
    
def getsubfolderfrompath(fullpath):
    mf=fullpath.split('\\')[-3]
    subf=fullpath.split('\\')[-2]
    return './'+mf+'/'+subf

def createfolder(foldername):
    import os
    if not os.path.exists(foldername):
        print('membuat folder baru : '+foldername)
        os.makedirs(foldername)
    
def saveFig(az,fname):
    createfolder(getfoldernamefrompath(fname))
    data = az.reshape((10,10)).T
    new_data = np.zeros(np.array(data.shape) * 10)
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j * 10: (j+1) * 10, k * 10: (k+1) * 10] = data[j, k]
    print('menyimpan gambar : '+fname)
    plt.imsave(str(fname),new_data)

def simpanGambar(stim,recon,fname):
    createfolder(getfoldernamefrompath(fname))
    plt.figure()
    sp1 = plt.subplot(131)
    sp1.axis('off')
    plt.title('Stimulus')
    sp2 = plt.subplot(132)
    sp2.axis('off')
    plt.title('Reconstruction')
    sp3 = plt.subplot(133)
    sp3.axis('off')
    plt.title('Binarized')
    sp1.imshow(stim.reshape((10,10)).T, cmap=plt.cm.gray,
               interpolation='nearest'),
    sp2.imshow(recon.reshape((10,10)).T, cmap=plt.cm.gray,
               interpolation='nearest'),
    sp3.imshow(np.reshape(recon > .5, (10, 10)).T, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.savefig(fname)

def rowtoimagematrix(rowimage):
    matriximage=rowimage.reshape((10,10)).T
    return matriximage

def plotting(label,pred,predm,fname):
    cols=['stimulus','rolly','miyawaki']
    fig, ax = plt.subplots(nrows=10, ncols=3,figsize=(5, 20))
    for axes, col in zip(ax[0], cols):
        axes.set_title(col)
    for row,fig,p,pm in zip(ax,label,pred,predm):
        row[0].axis('off')
        row[1].axis('off')
        row[2].axis('off')
        row[0].imshow(fig.reshape((10,10)).T, cmap=plt.cm.gray,
               interpolation='nearest'),
        row[1].imshow(p.reshape((10,10)).T, cmap=plt.cm.gray,
               interpolation='nearest'),
        row[2].imshow(pm.reshape((10, 10)).T, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.show()

def plotHasil(label,pred,predm,mse,msem,matfile,n,arch):
    fname1=getfigpath(matfile,'resultpict'+'\\'+arch,n)
    createfolder(getsubfolderfrompath(fname1))
    rows=['Stimulus','Rolly','Miyawaki']
    idx=list(range(1,len(mse)+1))
    fig, ax = plt.subplots(nrows=3, ncols=10,figsize=(15, 5))
    for axes, row in zip(ax[:,0], rows):
        axes.set_ylabel(row, rotation=90, size='large')
    for idn,col,fig in zip(idx,ax[0],label):
        col.set_yticklabels([])
        col.set_yticks([])
        col.set_xticklabels([])
        col.set_xticks([])
        col.imshow(fig.reshape((10,10)).T, cmap=plt.cm.gray,interpolation='nearest')
        col.set_title(idn)
    for col,p in zip(ax[1],pred):
        col.set_yticklabels([])
        col.set_yticks([])
        col.set_xticklabels([])
        col.set_xticks([])
        col.imshow(p.reshape((10,10)).T, cmap=plt.cm.gray,interpolation='nearest')
    for col,pm in zip(ax[2],predm):
        col.set_yticklabels([])
        col.set_yticks([])
        col.set_xticklabels([])
        col.set_xticks([])
        col.imshow(pm.reshape((10,10)).T, cmap=plt.cm.gray,interpolation='nearest')
    plt.suptitle(' Hasil Rekonstruksi Multilayer Perceptron : '+arch+', bagian ke-'+str(n), fontsize=16)
    # plt.show()
    plt.savefig(fname1)
    
    fname2=getfigpath(matfile,'resultmse'+'\\'+arch,n)
    createfolder(getsubfolderfrompath(fname2))
    fige, axe = plt.subplots(figsize=(15, 5))
    axe.plot(idx, mse, color = 'green', label = 'mse rolly')
    axe.plot(idx, msem, color = 'red', label = 'mse miyawaki')
    axe.legend(loc = 'lower left')
    axe.set_xticks(idx)
    # plt.show()
    plt.suptitle('Perbandingan Mean Suare Error', fontsize=16)
    plt.savefig(fname2)
    
    import PIL
    fnamegab=getfigpath(matfile,'results'+'\\'+arch,n)
    createfolder(getsubfolderfrompath(fnamegab))
    
    list_im = [fname1, fname2]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(fnamegab)

def plotDGMM(label,pred,predm,mse,msem,matfile,n,arch):
    fname1=getfigpath(matfile,'resultpict'+'\\'+arch,n)
    createfolder(getsubfolderfrompath(fname1))
    rows=['Stimulus','DGMM','Miyawaki']
    idx=list(range(1,len(mse)+1))
    fig, ax = plt.subplots(nrows=3, ncols=10,figsize=(15, 5))
    for axes, row in zip(ax[:,0], rows):
        axes.set_ylabel(row, rotation=90, size='large')
    for idn,col,fig in zip(idx,ax[0],label):
        col.set_yticklabels([])
        col.set_yticks([])
        col.set_xticklabels([])
        col.set_xticks([])
        col.imshow(fig.reshape((10,10)).T, cmap=plt.cm.gray,interpolation='nearest')
        col.set_title(idn)
    for col,p in zip(ax[1],pred):
        col.set_yticklabels([])
        col.set_yticks([])
        col.set_xticklabels([])
        col.set_xticks([])
        col.imshow(p.reshape((10,10)).T, cmap=plt.cm.gray,interpolation='nearest')
    for col,pm in zip(ax[2],predm):
        col.set_yticklabels([])
        col.set_yticks([])
        col.set_xticklabels([])
        col.set_xticks([])
        col.imshow(pm.reshape((10,10)).T, cmap=plt.cm.gray,interpolation='nearest')
    plt.suptitle(' Perbandingan Rekonstruksi '+arch+' dan Miyawaki, bagian ke-'+str(n), fontsize=16)
    # plt.show()
    plt.savefig(fname1)
    
    fname2=getfigpath(matfile,'resultmse'+'\\'+arch,n)
    createfolder(getsubfolderfrompath(fname2))
    fige, axe = plt.subplots(figsize=(15, 5))
    axe.plot(idx, mse, color = 'green', label = 'mse dgmm')
    axe.plot(idx, msem, color = 'red', label = 'mse miyawaki')
    axe.legend(loc = 'lower left')
    axe.set_xticks(idx)
    # plt.show()
    plt.suptitle('Perbandingan Mean Square Error', fontsize=16)
    plt.savefig(fname2)
    
    import PIL
    fnamegab=getfigpath(matfile,'results'+'\\'+arch,n)
    createfolder(getsubfolderfrompath(fnamegab))
    
    list_im = [fname1, fname2]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(fnamegab)
    

def delfirstCol(testlb):
    return np.delete(testlb,0,1)

def modelfolderpath(matfile):
    mpath='.\\'+matfile.split('_')[2]+'_'+matfile.split('_')[-2]+'\\'
    return mpath

def figfile(matfile,n):
    figfolderpath='.\\'+matfile.split('_')[2]+'_'+matfile.split('_')[-2]+'_fig'+'\\'+str(n)+'.png'
    return figfolderpath

def figrecfile(matfile,n):
    figfolderpath='.\\'+matfile.split('_')[2]+'_'+matfile.split('_')[-2]+'_figrec'+'\\'+str(n)+'.png'
    return figfolderpath

def getfigpath(matfile,suffix,n):
    import pathlib
    scriptDirectory = pathlib.Path().absolute()
    figfolderpath=str(scriptDirectory)+'\\'+matfile.split('_')[2]+'_'+matfile.split('_')[-2]+'_'+suffix+'\\'+str(n)+'.png'
    print('generate path gambar : '+figfolderpath)
    return figfolderpath

def msefilename(matfile,arch):
    import pathlib
    scriptDirectory = pathlib.Path().absolute()
    figfolderpath=str(scriptDirectory)+'\\'+matfile.split('_')[2]+'_'+matfile.split('_')[-2]+'_mse'+'\\'+str(arch)+'.csv'
    return figfolderpath

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        
def ubahkelistofchunks(mse,msem,pred,predm,label,n):
    mse=mse.tolist()
    msem=msem.tolist()
    lmse=list(divide_chunks(mse, n))
    lmsem=list(divide_chunks(msem, n))
    lpred=list(divide_chunks(pred, n))
    lpredm=list(divide_chunks(predm, n))
    llabel=list(divide_chunks(label, n))
    return lmse,lmsem,lpred,lpredm,llabel

def ssimscore(matrixgambar1,matrixgambar2):
    # SSIM: 1.0 is similar
    (score, diff) = structural_similarity(matrixgambar1, matrixgambar2, full=True)
    diff = (diff * 255).astype("uint8")
    # 6. You can print only the score if you want
    print("SSIM: {}".format(score))
    return score
    
def msescore(matrixgambar1,matrixgambar2):
    #mse 0:similar
    mse = np.mean( (matrixgambar1 - matrixgambar2) ** 2 )
    #mse = ((original - contrast)**2).mean(axis=1)
    return mse

def psnrscore(matrixgambar1,matrixgambar2):
    #psnr 1:similar
    mse = np.mean( (matrixgambar1 - matrixgambar2) ** 2 )
    if mse == 0:
        psnr = 100
    else:
        PIXEL_MAX = 255.0
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr/100

def corrscore(matrixgambar1,matrixgambar2):
    #1 maka similar
    cor = np.corrcoef(matrixgambar1.reshape(-1),matrixgambar2.reshape(-1))[0][1]
    return cor