import os
import sys

parent_dir = os.path.split(os.getcwd())[0]
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import random
import pickle

from utils import divide_texts, save_scores
from object_selection import ObjectSelection
from load_data import load_data
import matplotlib.pyplot as plt
from basic_factorization import BasicFactorization

# Parameters
data_directory = '../../data/generated-data-r-10-n-04/'
booking_file = '../../data/booking.csv'
users_file = '../../data/user.csv'
rating_thresholds = []
true_objects_indexes = [0, 1, 2, 3, 4, 5]
false_objects_indexes = [6, 7, 8, 9]

file_names = os.listdir(data_directory)
ids_vector = [int(name.split('-')[0]) for name in file_names]
categories_vector = [name.split('-')[1] for name in file_names]
ratings_vector = [int(name.split('.')[0].split('-')[2]) for name in file_names]
name_vector = [data_directory + name for name in file_names]

ratings_matrix, images_indexes_for_id, ids_indexes, users_matrix = load_data(data_directory, booking_file,
                                                                             users_file, rating_thresholds)

features, new_ratings_vector, new_categories_vector, new_ids_vector, new_paths_vector, text_indexes = divide_texts(
    name_vector, ratings_vector, categories_vector, ids_vector, n=10)

ratings_vector = new_ratings_vector
ids_vector = new_ids_vector

scores_auc = []
scores_rmse = []
for i in range(10):
    cv_results_file = '../results/cv-generated-data-r-10-n-04-knn-' + str(i) + '.csv'
    selection = BasicFactorization(show_selection_results=False, selection_algorithm='knn')
    selection.transform(ids=ids_vector, features=features, ratings=ratings_vector, users_ratings=ratings_matrix,
                        users=users_matrix, cv_results_file=cv_results_file, images_indexes=text_indexes,
                        true_objects_indexes=true_objects_indexes, false_objects_indexes=false_objects_indexes,
                        paths=name_vector, z_score=False)
    score_auc, score_rmse = selection.evaluate(evaluation_metric='auc')
    scores_auc.append(score_auc)
    scores_rmse.append(score_rmse)

scores_auc.sort()
plt.title('AUC za izbrane slike')
plt.plot(scores_auc)
plt.ylabel('AUC')
plt.show()

results_file = '../scores/generated-data-r-10-n-04-knn-auc.csv'
save_scores(scores_auc, results_file)
results_file = '../scores/generated-data-r-10-n-04-knn-rmse.csv'
save_scores(scores_rmse, results_file)