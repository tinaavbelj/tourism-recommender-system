import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_users_data(users_file):
    """
    Loads data for users to matrix (user row index = user_id - 1)

    :param users_file: path to csv file with data for users
    :returns:  users matrix
    """
    data_users = pd.read_csv(users_file, sep=';')
    user_ids = data_users['Id']
    user_group = data_users['PersonGroup']
    user_gender = data_users['Gender']
    user_age = data_users['AgeRange']
    user_tourist_type = data_users['TouristType']
    user_country = data_users['Country']

    users_array = []
    for i in range(len(user_ids)):
        new_user_data = [user_group[i], user_gender[i], user_age[i], user_tourist_type[i], user_country[i]]
        users_array.append(new_user_data)

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(users_array)
    users_matrix = encoder.transform(users_array).toarray()
    users_matrix = np.array(users_matrix)

    return users_matrix


def load_ratings_data(booking_file, n_experiences, ids_indexes, rating_thresholds):
    """
    Loads data for ratings to matrix

    :param booking_file: path to csv file with ratings
    :param n_experiences: number of all experiences
    :param ids_indexes: keys - experience ids, values - indexes for rows in matrix
    :param rating_thresholds: array of (min) thresholds for ratings
    :returns:  ratings matrix
    """
    data_booking = pd.read_csv(booking_file, sep=';')
    experience_ids_booking = list(data_booking['UserId'])
    user_ids = list(data_booking['Rating'])
    experience_ratings = list(data_booking['TimeOfDay'])

    # Ratings matrix
    unique_user_ids = list(set(user_ids))
    n_users = len(unique_user_ids)
    ratings_matrix = np.zeros(shape=(n_users, n_experiences))

    for current_user_id in unique_user_ids:
        # Find indexes for this user
        indexes = []
        for i, user_id in enumerate(user_ids):
            if user_id == current_user_id:
                indexes.append(i)

        # Add data for current user to ratings matrix
        new_user_id = current_user_id - 1

        for index in indexes:
            # If this experience exists in generated pictures
            if int(experience_ids_booking[index]) in ids_indexes.keys():
                new_experience_id = ids_indexes[experience_ids_booking[index]]
                #right_threshold_index = 0
                #for t_index, threshold in enumerate(rating_thresholds):
                #    if experience_ratings[index] >= threshold:
                #        right_threshold_index = t_index
                #new_rating = right_threshold_index + 1
                new_rating = experience_ratings[index]
                ratings_matrix[new_user_id][new_experience_id] = new_rating

    return ratings_matrix


def average_images_matrix(images_indexes_for_id, n_experiences, ids_indexes, features):
    """
    Creates matrix with average of all images for experience

    :param images_indexes_for_id: object keys - experience ids, values - arrays of images indexes for id
    :param n_experiences: number of all experiences
    :param ids_indexes: keys - experience ids, values - indexes for rows in matrix
    :param features: features for all images
    :returns:  images matrix
    """
    images_matrix = np.zeros(shape=(2048, n_experiences))

    for img_id in images_indexes_for_id.keys():
        new_id = ids_indexes[img_id]

        image_vector = np.zeros(2048)

        for im in range(len(images_indexes_for_id[img_id])):
            image_vector += features[images_indexes_for_id[img_id][im]]

        image_vector = image_vector / len(images_indexes_for_id[img_id])

        images_matrix[:, new_id] = image_vector

    return images_matrix


def calculate_binary_ratings(ratings_matrix):
    """
    Calculates experience ratings from average user ratings

    :param ratings_matrix: matrix of user ratings
    :returns:  binary ratings matrix
    """

    n_users = ratings_matrix.shape[0]
    n_experiences = ratings_matrix.shape[1]

    # Calculate average for each user (only non-zero values)
    users_avg = []
    for i in range(n_users):
        ratings_sum = 0
        n = 0
        for j in range(n_experiences):
            if ratings_matrix[i, j] != 0:
                ratings_sum += ratings_matrix[i, j]
                n += 1
        average = ratings_sum / n
        users_avg.append(average)

    # Subtract user average from each rating
    average_ratings_matrix = np.zeros(ratings_matrix.shape)
    for i in range(n_users):
        for j in range(n_experiences):
            if ratings_matrix[i, j] != 0:
                new_rating = ratings_matrix[i, j] - users_avg[i]
                average_ratings_matrix[i, j] = new_rating

    # Define new binary rating based on average rating
    binary_ratings_matrix = np.zeros(ratings_matrix.shape)
    for i in range(n_users):
        for j in range(n_experiences):
            if ratings_matrix[i, j] != 0:
                if average_ratings_matrix[i, j] > 0:
                    binary_ratings_matrix[i, j] = 2
                else:
                    binary_ratings_matrix[i, j] = 1

    return binary_ratings_matrix


def load_data(data_directory, booking_file, users_file, rating_thresholds):
    file_names = os.listdir(data_directory)
    img_ids_vector = [int(name.split('-')[0]) for name in file_names]

    # Ids to column indexes
    new_column_index = 0
    ids_indexes = {}
    for i in sorted(list(set(img_ids_vector))):
        ids_indexes[i] = new_column_index
        new_column_index += 1

    # Dictionary of images indexes for experience id
    images_indexes_for_id = {}
    unique_ids = list(set(img_ids_vector))
    for i in unique_ids:
        images_indexes_for_id[i] = []

    for current_id in unique_ids:
        for index, image_id in enumerate(img_ids_vector):
            # Find image indexes for id
            if image_id == current_id:
                images_indexes_for_id[image_id].append(index)

    n_experiences = len(sorted(list(set(img_ids_vector))))

    # Matrix with average of all images for experience as columns
    # average_images_matrix(images_indexes_for_id, n_experiences, ids_indexes, features)

    ratings_matrix = load_ratings_data(booking_file, n_experiences, ids_indexes, rating_thresholds)
    users_matrix = load_users_data(users_file)
    binary_ratings_matrix = calculate_binary_ratings(ratings_matrix)

    return binary_ratings_matrix, images_indexes_for_id, ids_indexes, users_matrix
