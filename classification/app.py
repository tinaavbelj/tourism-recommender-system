from classification import Classification
from classification_text import ClassificationText
from preselection import Preselection
from preselection2 import Preselection2


def main():
    # Preselection example

    # Parameters
    data_directory = '../data/generated-data-r-2-n-3-3'
    features_path = '../data/features-generated-data-r-2-n-3-3'
    results_file = '/results-preselection/generated-data-r-2-n-3-3.csv'
    true_objects_indexes = [0, 1, 2]
    false_objects_indexes = [3, 4, 5]

    preselection = Preselection(data_directory, features_path, true_objects_indexes, false_objects_indexes)
    preselection.transform(results_file=results_file)
    #preselection.evaluate()

    exit()

    # Texts example

    # Parameters
    data_directory = '../data/data-real-r-3-text'
    results_file = '/results-text/all_knn.csv'

    classification = ClassificationText(data_directory, algorithm='knn', feature_agglomeration=False, selection='none')
    classification.transform(results_file=results_file)
    classification.evaluate()

    exit()

    # Images example

    # Parameters
    data_directory = '../data/data-real-r-3'
    features_path = '../data/features-data-real-r-3'
    results_file = '/results/kmeans_knn.csv'

    classification = Classification(data_directory, features_path, algorithm='knn', feature_agglomeration=False,
                                    selection='kmeans')
    classification.transform(results_file=results_file)
    classification.evaluate()


if __name__ == '__main__':
    main()