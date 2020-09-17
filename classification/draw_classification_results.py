import seaborn as sns; sns.set()
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams
import os
import sys

parent_dir = os.path.split(os.getcwd())[0]
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from classification import Classification


def main():
    # Parameters
    data_directory = '../data/data-real-r-3'
    features_path = '../data/features-data-real-r-3'
    results_file = './results/silhouette_rf.csv'

    classification = Classification(data_directory, features_path, algorithm='knn', feature_agglomeration=True)
    classification.transform(results_file=results_file)
    classification.evaluate(text_to_add='RF, silhueta')

    exit()

    # Parameters
    data_directory = '../data/data-real-r-3'
    features_path = '../data/features-data-real-r-3'
    results_file = './results-text/all_rf.csv'

    classification = Classification(data_directory, features_path, algorithm='knn', feature_agglomeration=True)
    classification.transform(results_file=results_file)
    classification.evaluate(text_to_add='RF')


if __name__ == '__main__':
    main()
