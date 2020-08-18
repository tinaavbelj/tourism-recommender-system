import os
import random

from utils import get_features
from object_selection import ObjectSelection
from load_data import load_data
from basic_factorization import BasicFactorization


def main():
    # Parameters
    #data_directory = '../data/experience-6/'
    #features_path = '../data/features-experience-6'
    data_directory = '../data/generated-data-nr-2-n-8-2/'
    features_path = '../data/features-generated-data-nr-2-n-8-2'
    booking_file = '../data/booking.csv'
    users_file = '../data/user.csv'
    cv_results_file = 'results/cv-generated-data-r-2-n-8-2-knn.csv'
    true_objects_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    false_objects_indexes = [8, 9]

    #file_to_delete = data_directory + '.DS_Store'
    #os.remove(file_to_delete)

    false_objects_indexes = [8, 9]

    file_names = os.listdir(data_directory)
    img_ids_vector = [int(name.split('-')[0]) for name in file_names]
    ratings_vector = [int(name.split('-')[-2]) for name in file_names]
    name_vector = [data_directory + name for name in file_names]
    images_indexes = [name.split('-')[3].split('.')[0] for name in file_names]
    rating_thresholds = [1, 2]

    ratings_matrix, images_indexes_for_id, ids_indexes, users_matrix = load_data(data_directory, booking_file,
                                                                                 users_file, rating_thresholds)

    features = get_features(features_path, name_vector)

    cv_results_file = './results/cv-generated-data-r-2-n-8-2-knn.csv'

    selection = BasicFactorization(show_selection_results=True, selection_algorithm='random')
    selection.transform(ids=img_ids_vector, features=features, ratings=ratings_vector, users_ratings=ratings_matrix,
                        users=users_matrix, cv_results_file=cv_results_file, images_indexes=images_indexes,
                        true_objects_indexes=true_objects_indexes, false_objects_indexes=false_objects_indexes,
                        paths=name_vector)
    selection.evaluate(evaluation_metric='auc')

    exit()

    selection = ObjectSelection(show_selection_results=True, selection_algorithm='knn')
    selection.transform(ids=img_ids_vector, features=features, ratings=ratings_vector, users_ratings=ratings_matrix,
                        users=users_matrix, cv_results_file=cv_results_file, images_indexes=images_indexes,
                        true_objects_indexes=true_objects_indexes, false_objects_indexes=false_objects_indexes,
                        paths=name_vector)
    selection.evaluate(evaluation_metric='auc')


if __name__ == '__main__':
    main()