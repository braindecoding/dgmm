from tensorflow.keras.layers import Input, Dense, Lambda, Flatten,Reshape
from tensorflow.keras.layers import Conv2D,Conv2DTranspose
from tensorflow.keras import backend

# In[]: Building the architechture

def sampling(args):
    #is a function that applies the reparametrization trick, 
    #a technique that allows us to backpropagate gradients through the stochastic latent variable.
    #The Lambda layer is used to apply this function to the inputs.
    Z_mu, Z_lsgms, K = args
    epsilon = backend.random_normal(shape=(backend.shape(Z_mu)[0], K), mean=0., stddev=1.0)
    
    return Z_mu + backend.exp(Z_lsgms) * epsilon
    
def encoder(X,D2,img_chns,filters,num_conv,intermediate_dim,K):
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
    # fungsi lambda berguna untuk pura2 menambah layer dengan hanya operasi perhitungan saja 
    # Sebagai contoh sederhana, misalkan Anda memiliki tensor masukan dan Anda ingin mengkuadratkannya.
    Z = Lambda(sampling, output_shape=(K,))([Z_mu, Z_lsgms,K])
    return Z,Z_lsgms,Z_mu

def decoderars(intermediate_dim,filters,batch_size,num_conv,img_chns):
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * 14 * 14, activation='relu')

    if backend.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 14, 14)
    else:
        output_shape = (batch_size, 14, 14, filters)

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
    # if backend.image_data_format() == 'channels_first':
    #     output_shape = (batch_size, filters, 29, 29)
    # else:
    #     output_shape = (batch_size, 29, 29, filters)
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash_mu = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')

    decoder_mean_squash_lsgms= Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='tanh')
    return decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms

def decoders(Z,decoder_hid,decoder_upsample,decoder_reshape,decoder_deconv_1,decoder_deconv_2,decoder_deconv_3_upsamp,decoder_mean_squash_mu,decoder_mean_squash_lsgms):
    hid_decoded = decoder_hid(Z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    X_mu = decoder_mean_squash_mu (x_decoded_relu)
    X_lsgms = decoder_mean_squash_lsgms (x_decoded_relu)
    return X_mu,X_lsgms

def decoder(Z,intermediate_dim,filters,batch_size,num_conv,img_chns):
    hid_decoded = Dense(intermediate_dim, activation='relu')(Z)
    up_decoded = Dense(filters * 14 * 14, activation='relu')(hid_decoded)
    if backend.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 14, 14)
    else:
        output_shape = (batch_size, 14, 14, filters)
    reshape_decoded = Reshape(output_shape[1:])(up_decoded)
    deconv_1_decoded = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')(reshape_decoded)
    deconv_2_decoded = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')(deconv_1_decoded)
    if backend.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 29, 29)
    else:
        output_shape = (batch_size, 29, 29, filters)
    x_decoded_relu = Conv2DTranspose(filters,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='valid',
                                    activation='relu')(deconv_2_decoded)
    X_mu = Conv2D(img_chns,
                kernel_size=2,
                padding='valid',
                activation='sigmoid') (x_decoded_relu)
    X_lsgms = Conv2D(img_chns,
                    kernel_size=2,
                    padding='valid',
                    activation='tanh') (x_decoded_relu)
    return X_mu,X_lsgms



def imagereconstruct(K,intermediate_dim,filters,batch_size,num_conv,img_chns):
    Z_predict = Input(shape=(K,))
    _hid_decoded = Dense(intermediate_dim, activation='relu')(Z_predict)
    _up_decoded = Dense(filters * 14 * 14, activation='relu')(_hid_decoded)
    if backend.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 14, 14)
    else:
        output_shape = (batch_size, 14, 14, filters)
    _reshape_decoded = Reshape(output_shape[1:])(_up_decoded)
    _deconv_1_decoded = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')(_reshape_decoded)
    _deconv_2_decoded = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')(_deconv_1_decoded)
    _x_decoded_relu = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')(_deconv_2_decoded)
    X_mu_predict = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')(_x_decoded_relu)
    return X_mu_predict,Z_predict