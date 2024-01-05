import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import FrechetInceptionDistance
import numpy as np

# Load the Inception-v3 model pre-trained on ImageNet
base_model = InceptionV3(weights='imagenet')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Function to preprocess images and extract features
def preprocess_and_extract_features(images):
    images = preprocess_input(images)
    features = feature_extractor.predict(images)
    return features

# Function to calculate FID using stim and rec as generated images
def calculate_fid(real_images, generated_images):
    real_features = preprocess_and_extract_features(real_images)
    generated_features = preprocess_and_extract_features(generated_images)

    # Assuming you have a function to calculate FID using the extracted features
    fid_value = calculate_fid_with_features(real_features, generated_features)

    return fid_value

def calculate_fid_with_features(real_features, generated_features):
    fid_metric = FrechetInceptionDistance()
    fid_metric.update_state(real_features, generated_features)
    fid_value = fid_metric.result().numpy()
    return fid_value