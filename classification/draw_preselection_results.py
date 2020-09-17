import os
import sys

from preselection import Preselection

# Preselection example

# Parameters
#data_directory = '../data/old/generated-data-r-2-n-8-2'
#features_path = '../data/old/features-generated-data-r-2-n-8-2'
data_directory = '../data/generated-data-categories'
features_path = '../data/features-generated-data-categories'
results_file = './results-preselection/generated-data-r-2-n-8-2.csv'
true_objects_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
false_objects_indexes = [8, 9]

#preselection = Preselection(data_directory, features_path, true_objects_indexes, false_objects_indexes)
#preselection.transform(results_file=results_file)

# Parameters
data_directory = '../data/old/generated-data-r-2-n-6-4'
features_path = '../data/old/features-generated-data-r-2-n-6-4'
results_file = './results-preselection/generated-data-r-2-n-6-4.csv'
true_objects_indexes = [0, 1, 2, 3, 4, 5]
false_objects_indexes = [6, 7, 8, 9]

preselection = Preselection(data_directory, features_path, true_objects_indexes, false_objects_indexes)
preselection.transform(results_file=results_file)