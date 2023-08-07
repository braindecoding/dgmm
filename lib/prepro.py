from scipy.io import loadmat

from sklearn import preprocessing

def getXY(filepath,resolution):
    handwriten_69=loadmat(filepath)
    #ini fmri 10 test 90 train satu baris berisi 3092
    Y_train = handwriten_69['fmriTrn'].astype('float32')
    Y_test = handwriten_69['fmriTest'].astype('float32')
    
    # ini stimulus semua
    X_train = handwriten_69['stimTrn']#90 gambar dalam baris isi per baris 784 kolom
    X_test = handwriten_69['stimTest']#10 gambar dalam baris isi 784 kolom
    
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    
    #channel di depan
    #X_train = X_train.reshape([X_train.shape[0], 1, resolution, resolution])
    #X_test = X_test.reshape([X_test.shape[0], 1, resolution, resolution])
    #channel di belakang(edit rolly) 1 artinya grayscale
    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])
    # In[]: Normlization sinyal fMRI, min max agar nilainya hanya antara 0 sd 1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)
    return X_train,X_test,Y_train,Y_test