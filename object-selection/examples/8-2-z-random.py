import os
import sys
parent_dir = os.path.split(os.getcwd())[0]
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import random
from sklearn.cluster import FeatureAgglomeration

from utils import get_features, save_scores
from object_selection import ObjectSelection
from load_data import load_data
from basic_factorization_nimfa import BasicFactorizationNmf
from basic_factorization import BasicFactorization


def main():

    # Parameters
    data_directory = '../../data/generated-data-r-10-n-8-2/'
    features_path = '../../data/features-generated-data-r-10-n-8-2'
    booking_file = '../../data/booking.csv'
    users_file = '../../data/user.csv'
    rating_thresholds = []
    true_objects_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    false_objects_indexes = [8, 9]

    file_names = os.listdir(data_directory)
    img_ids_vector = [int(name.split('-')[0]) for name in file_names]
    ratings_vector = [int(name.split('-')[-2]) for name in file_names]
    name_vector = [data_directory + name for name in file_names]
    images_indexes = [name.split('-')[3].split('.')[0] for name in file_names]

    ratings_matrix, images_indexes_for_id, ids_indexes, users_matrix = load_data(data_directory, booking_file,
                                                                                 users_file, rating_thresholds)

    features = get_features(features_path, name_vector)

    scores_auc = []
    scores_rmse = []
    for i in range(10):
        cv_results_file = '../results/cv-generated-data-r-10-n-8-2-z-random-' + str(i) + '.csv'
        selection = ObjectSelection(show_selection_results=False, selection_algorithm='random')
        selection.transform(ids=img_ids_vector, features=features, ratings=ratings_vector, users_ratings=ratings_matrix,
                            users=users_matrix, cv_results_file=cv_results_file, images_indexes=images_indexes,
                            true_objects_indexes=true_objects_indexes, false_objects_indexes=false_objects_indexes,
                            paths=name_vector, z_score=True)
        selection.evaluate(evaluation_metric='auc')
        selection.evaluate(evaluation_metric='rmse')
        print('\n\n-----\n\n')
        score_auc, score_rmse = selection.evaluate(evaluation_metric='auc')
        scores_auc.append(score_auc)
        scores_rmse.append(score_rmse)

    results_file = '../scores/generated-data-r-10-n-8-2-z-random-auc.csv'
    save_scores(scores_auc, results_file)
    results_file = '../scores/generated-data-r-10-n-8-2-z-random-rmse.csv'
    save_scores(scores_rmse, results_file)


if __name__ == '__main__':
    main()