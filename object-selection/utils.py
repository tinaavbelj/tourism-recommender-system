import tensorflow as tf
import numpy as np
import os
import pickle
from gensim.models import Word2Vec
import pandas as pd
import gensim


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


def text_to_list(file):
    """
    Read text from the file to a list

    :param file: path to text
    :returns:  list of words in the text
    """
    with open(file, 'r') as f:
        text = f.read()
    words = text.split()
    return words


def text_to_vector(text, model):
    """
    Calculate vector for text as average of vectors of all words in the text

    :param text: List of words in a text
    :param model: Word2Vec model made on all texts in the dataset
    :returns:  vector representing text
    """
    vector = np.zeros(100)
    num_words = 0
    for word in text:
        try:
            vector = np.add(vector, model[word])
            num_words += 1
        except:
            pass
    return vector / np.sqrt(vector.dot(vector))


def texts_to_vectors(texts_paths):
    """
    Calculate matix of features for texts_paths (array of paths)

    :param texts_paths: List of paths to texts
    :returns:  features
    """

    all_texts = []
    for name in texts_paths:
        current_text = text_to_list(name)
        all_texts.append(current_text)

    model = Word2Vec(all_texts, size=100, min_count=1)

    features = []
    for text in all_texts:
        features.append(text_to_vector(text, model))

    features = np.asarray(features)

    return features


def divide_texts(paths, ratings_vector, categories_vector, ids_vector, n=15, model='', features_directory=''):
    """
    Divide texts in texts of length n

    :param paths: List of paths to texts
    :param ratings_vector: Array of ratings
    :param categories_vector: Array of categories
    :param ids_vector: Array of ids
    :param n: length of texts
    :returns:  features, new_ratings_vector, new_categories_vector, new_ids_vector, new_paths_vector
    """
    all_texts = []
    new_ratings_vector = []
    new_categories_vector = []
    new_ids_vector = []
    new_paths_vector = []
    text_indexes = []

    for i, path in enumerate(paths):
        text = text_to_list(path)
        part_length = round(len(text) / n)
        text = np.array(text)
        parts = list(np.array_split(text, part_length))

        for index, p in enumerate(parts):
            p = p.tolist()
            all_texts.append(p)
            new_ratings_vector.append(ratings_vector[i])
            new_categories_vector.append(categories_vector[i])
            new_ids_vector.append(ids_vector[i])
            new_paths_vector.append(paths[i])
            text_indexes.append(index)

    if model == '':
        model = Word2Vec(all_texts, size=100, min_count=1)

    features = np.zeros((len(all_texts), 100))
    if features_directory == '':
        for i, text in enumerate(all_texts):
            features[i] = text_to_vector(text, model)
    else:
        i = 0
        for name in list(set(new_paths_vector)):
            features_name = name.split('/')[-1].split('.')[0] + '.csv'
            columns = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']
            data = pd.read_csv(features_directory + '/' + features_name, sep=',',
                               names=columns)

            for c in columns:
                features[i] = list(data[c])[1:]
                i = i + 1

    return features, new_ratings_vector, new_categories_vector, new_ids_vector, new_paths_vector, text_indexes


def save_scores(scores, results_file):
    """
    Save scores to a file

    :param scores: array of scores
    :param results_file: path for predictions
    """
    # Save to csv file
    data = {'scores': scores}
    df = pd.DataFrame(data, columns=['scores'])
    df.to_csv(results_file)
