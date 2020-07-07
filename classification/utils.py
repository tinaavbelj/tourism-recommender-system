import tensorflow as tf
import numpy as np
import os
import pickle


def load_image(image_path):
    """
    Load image with tensorflow

    :param image_path: path to image
    :returns:  image
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def get_feature_extraction_model():
    """
    Create feature extraction model for penultimate layer of InceptionV3

    :returns: extraction model
    """
    image_model = tf.keras.applications.InceptionV3(input_shape=(299, 299, 3), include_top=False,
                                                    weights='imagenet', pooling='avg')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model


def calculate_features(img_name_vector):
    """
    Calculate vectors of features for all images

    :param img_name_vector: List of paths to images
    :returns: features (np array with vectors representing image features in each row)
    """
    image_dataset = tf.data.Dataset.from_tensor_slices(img_name_vector)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    print('\nCalculate features\n')

    model = get_feature_extraction_model()
    features = np.empty(shape=(0, 2048))
    for img in image_dataset:
        batch_features = model(img)
        features = np.vstack((features, batch_features))

    return features


def get_features(features_path, images_paths):
    """
    Return existing features or calculate features

    :param features_path: path to features
    :param images_paths: array of paths for all images
    :returns: features (np array with vectors representing image features in each row)
    """
    if not os.path.exists(features_path):
        features = calculate_features(images_paths)
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
    else:
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
    return features
