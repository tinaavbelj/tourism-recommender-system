from classification import Classification
from classification_text import ClassificationText


def main():
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