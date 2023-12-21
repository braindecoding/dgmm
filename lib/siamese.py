import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
from tensorflow.keras import backend as K


def mse_score(sample1, sample2):
    """Calculate the Mean Squared Error between two samples."""
    return np.mean(np.square(sample1 - sample2))

def create_pairs_with_mse(Y_train, Y_test, threshold):
    """
    Create pairs of samples from Y_train and Y_test based on MSE.

    Parameters:
    Y_train (np.array): Training data.
    Y_test (np.array): Testing data.
    threshold (float): Threshold for determining if a pair is considered similar (below threshold) or dissimilar (above threshold).

    Returns:
    pairs (np.array): Array containing pairs of samples.
    labels (np.array): Array containing labels for each pair (1 for similar, 0 for dissimilar).
    """
    pairs = []
    labels = []

    for test_sample in Y_test:
        # Initialize a variable to store the minimum MSE and corresponding train sample
        min_mse = float('inf')
        min_mse_sample = None

        # Iterate through all samples in Y_train
        for train_sample in Y_train:
            mse = mse_score(test_sample, train_sample)
            if mse < min_mse:
                min_mse = mse
                min_mse_sample = train_sample

        # Check if the minimum MSE is below the threshold
        if min_mse < threshold:
            # Pair is considered similar
            pairs.append([test_sample, min_mse_sample])
            labels.append(1)
        else:
            # Pair is considered dissimilar
            pairs.append([test_sample, min_mse_sample])
            labels.append(0)

    return np.array(pairs), np.array(labels)

def create_pairs(Y_train, Y_test):
    return create_pairs_with_mse(Y_train, Y_test, 0.75)

from tensorflow.keras.layers import Conv1D, MaxPooling1D

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape

def initialize_base_network(input_shape):
    """
    Initialize the base network for the Siamese Network.

    Parameters:
    input_shape (tuple): Shape of the input data (length, channels).

    Returns:
    Model: A Keras model representing the base network.
    """
    input = Input(shape=input_shape)
    # Add a channel dimension to the input
    x = Reshape((input_shape[0], 1))(input)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def initialize_base_networkold(input_shape):
    """
    Initialize the base network for the Siamese Network.

    Parameters:
    input_shape (tuple): Shape of the input data (height, width, channels).

    Returns:
    Model: A Keras model representing the base network.
    """
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def euclidean_distance(vects):
    """
    Compute Euclidean distance between two vectors.
    
    Parameters:
    vects (list/tuple): A list or tuple of two tensors.
    
    Returns:
    Tensor: A tensor containing the Euclidean distance 
            between the vectors in 'vects'.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def S(k, t, Y_train, Y_test):
    # Create pairs of samples
    pairs, labels = create_pairs(Y_train, Y_test)

    # Assuming Y_train and Y_test are one-dimensional arrays with length 3092
    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], 1))

    # Define the Siamese Network
    input_shape = Y_train.shape[1:]  # Exclude batch size
    base_network = initialize_base_network(input_shape)

    # Reshape pairs for Conv1D input
    pairs = [pair.reshape((pair.shape[0], pair.shape[1], 1)) for pair in pairs[:, :2]]

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    # Train the model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit([np.array(pairs)[:, 0], np.array(pairs)[:, 1]], labels, epochs=int(t), batch_size=int(k))

    # Use the model to compute similarity/distance scores
    n_test = Y_test.shape[0]
    n_train = Y_train.shape[0]

    distances = np.zeros((n_train, n_test))
    for i in range(n_test):
        for j in range(n_train):
            distances[j, i] = model.predict([np.expand_dims(Y_test[i], axis=0), np.expand_dims(Y_train[j], axis=0)])

    return distances




def Sold(k, t, Y_train, Y_test):
    # Membuat pasangan sampel
    pairs, labels = create_pairs(Y_train, Y_test)

    # Mendefinisikan Siamese Network
    input_shape = Y_train[0].shape
    base_network = initialize_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)

    # Melatih model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=t, batch_size=k)

    # Menggunakan model untuk menghitung skor kesamaan/jarak
    n_test = Y_test.shape[0]
    n_train = Y_train.shape[0]

    distances = np.zeros((n_train, n_test))
    for i in range(n_test):
        for j in range(n_train):
            distances[j, i] = model.predict([np.expand_dims(Y_test[i], axis=0), np.expand_dims(Y_train[j], axis=0)])

    return distances
