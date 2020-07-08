import os
import random

from utils import get_features
from object_selection import ObjectSelection
from load_data import load_data


def main():
    # Parameters
    #data_directory = '../data/experience-6/'
    #features_path = '../data/features-experience-6'
    data_directory = '../data/experience-6-mix/'
    features_path = '../data/features-experience-6-mix'
    booking_file = '../data/booking.csv'
    users_file = '../data/user.csv'

    #file_to_delete = data_directory + '.DS_Store'
    #os.remove(file_to_delete)

    file_names = os.listdir(data_directory)
    img_ids_vector = [int(name.split('-')[0]) for name in file_names]
    ratings_vector = [int(name.split('-')[-2]) for name in file_names]
    name_vector = [data_directory + name for name in file_names]
    rating_thresholds = [1, 7]

    ratings_matrix, images_indexes_for_id, ids_indexes, users_matrix = load_data(data_directory, booking_file, users_file, rating_thresholds)

    new_ratings = []
    for r in ratings_vector:
        if r == 1:
            new_ratings.append(1)
        if r == 3:
            new_ratings.append(2)
        else:
            c = random.choice([1, 2])
            if c == 1:
                new_ratings.append(1)
            else:
                new_ratings.append(2)

    features = get_features(features_path, name_vector)

    selection = ObjectSelection(show_selection_results=True, selection_algorithm='random')
    selection.transform(ids=img_ids_vector, features=features, ratings=ratings_vector, users_ratings=ratings_matrix, users=users_matrix)
    selection.evaluate(evaluation_metric='auc')


if __name__ == '__main__':
    main()