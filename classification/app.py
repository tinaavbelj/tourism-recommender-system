from classification import Classification


def main():
    # Parameters
    data_directory = '../data/experience-6-mix'
    features_path = '../data/features-experience-6-mix'
    results_file = './results/knn'

    classification = Classification(data_directory, features_path, algorithm='knn', feature_agglomeration=False)
    classification.transform(results_file=results_file)
    classification.evaluate()


if __name__ == '__main__':
    main()