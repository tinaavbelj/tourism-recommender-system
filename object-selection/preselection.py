import seaborn as sns; sns.set()
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import KFold
import numpy as np
import copy

E = 0.001
MAX_IT = 10


class Preselection:
    def __init__(self, true_objects_indexes, false_objects_indexes):
        self.true_objects_indexes = true_objects_indexes
        self.false_objects_indexes = false_objects_indexes

        self.predicted_ratings_object = {}
        self.true_ratings_object = {}
        self.predicted_ratings_vector = []
        self.true_ratings_vector = []
        self.unique_ratings = []
        self.images_paths_object = {}
        self.ids_object = {}

    def load_results(self, results_file):
        """
        Load results

        :param results_file: path to previously computed predictions
        """
        predicted_ratings_object = {}
        true_ratings_object = {}
        predicted_ratings_vector = []
        true_ratings_vector = []

        data = pd.read_csv(results_file, sep=',', names=['provider_id', 'image_file', 'true_rating', 'predicted_rating'])
        ids = list(data['provider_id'])[1:]
        true_ratings = list(data['true_rating'])[1:]
        predicted_ratings = list(data['predicted_rating'])[1:]

        unique_ids = list(set(ids))

        for provider_id in unique_ids:
            predicted_ratings_object[provider_id] = []
            true_ratings_object[provider_id] = []

        for index, provider_id in enumerate(ids):
            predicted_ratings_object[provider_id].append(predicted_ratings[index])
            true_ratings_object[provider_id].append(true_ratings[index])
            predicted_ratings_vector.append(predicted_ratings[index])
            true_ratings_vector.append(true_ratings[index])

        self.predicted_ratings_object = predicted_ratings_object
        self.true_ratings_object = true_ratings_object
        self.predicted_ratings_vector = predicted_ratings_vector
        self.true_ratings_vector = true_ratings_vector

    def save_results(self, results_file):
        """
        Save predictions to a file

        :param results_file: path for predictions
        """
        # Save to csv file
        csv_provider_ids = []
        csv_image_files = []
        csv_true_ratings = []
        csv_predicted_ratings = []

        for index, key in enumerate(self.true_ratings_object.keys()):
            for el in self.predicted_ratings_object[key]:
                csv_predicted_ratings.append(el)
            for el in self.true_ratings_object[key]:
                csv_true_ratings.append(el)
            for el in self.images_paths_object[key]:
                csv_image_files.append(el)
            for el in self.ids_object[key]:
                csv_provider_ids.append(el)

        data = {'provider_id': csv_provider_ids,
                'image_file': csv_image_files,
                'true_rating': csv_true_ratings,
                'predicted_rating': csv_predicted_ratings}
        df = pd.DataFrame(data, columns=['provider_id', 'image_file', 'true_rating', 'predicted_rating'])
        df.to_csv(results_file)

    def split_providers(self, ids_vector, ratings_vector, features, test_ratio=0.2):
        # Divide provider ids to train and test
        unique_ids = list(set(ids_vector))
        random.shuffle(unique_ids)
        n = round(test_ratio * len(unique_ids))
        test_ids = unique_ids[0:n]
        train_ids = unique_ids[n:]

        # Find indexes for train and test set
        train_indexes = []
        test_indexes = []
        for index, id_ in enumerate(ids_vector):
            if id_ in test_ids:
                test_indexes.append(index)
            else:
                train_indexes.append(index)

        # Test and train data
        train_X = features[train_indexes, :]
        test_X = features[test_indexes, :]
        train_y = [x for index, x in enumerate(ratings_vector) if index in train_indexes]
        test_y = [x for index, x in enumerate(ratings_vector) if index in test_indexes]

        return train_X, test_X, train_y, test_y, train_ids, test_ids, train_indexes, test_indexes

    def transform(self, paths, ids_vector, ratings_vector, features, results_file=''):
        """
        Classify images for each provider and save predictions

        :param results_file: path to previously computed predictions
        """
        print('\nPreselection\n')
        #if path.exists(results_file):
        #    self.load_results(results_file)
        #    return

        images_paths = paths

        # Split data
        #train_X, test_X, train_y, test_y, train_ids, test_ids, train_indexes, test_indexes = self.split_providers(ids_vector, ratings_vector, features)

        current_it = 0
        X = np.array(features)
        print(X.shape)
        y = ratings_vector
        ids = ids_vector
        #train_images_indexes = [x for index, x in enumerate(images_indexes) if index in train_indexes]

        ca = 0
        ca_new = 1
        while current_it < MAX_IT and abs(ca - ca_new) > E:
            correct_indexes = []
            kf = KFold(n_splits=3)
            ca = ca_new
            kfolds_ca = []
            for train_index, test_index in kf.split(X):
                current_train_X, current_test_X = X[train_index], X[test_index]
                current_train_y = [x for index, x in enumerate(y) if index in train_index]
                current_test_y = [x for index, x in enumerate(y) if index in test_index]
                #current_images_indexes = [x for index, x in enumerate(train_images_indexes) if index in test_index]

                model = KNeighborsClassifier()
                model.fit(current_train_X, current_train_y)
                predicted = model.predict(current_test_X)
                current_ca = accuracy_score(current_test_y, predicted)
                kfolds_ca.append(current_ca)
                correct_indexes = correct_indexes + [test_index[index] for index, p in enumerate(predicted) if p == current_test_y[index]]

            ca_new = sum(kfolds_ca) / len(kfolds_ca)
            print('-')
            print(len(correct_indexes))
            print(ca_new)
            print('-')

            previous_ids = copy.deepcopy(ids)
            ids = [x for index, x in enumerate(ids) if index in correct_indexes]
            # Check if any of the provider's images were all left out
            missing_ids = [x for x in previous_ids if x not in ids]
            missing_ids = list(set(missing_ids))
            # Add one image for the left out provider
            missing_indexes = []
            for missing_id in missing_ids:
                # Find image from previous iteration
                index_to_add = previous_ids.index(missing_id)
                missing_indexes.append(index_to_add)

            correct_indexes = correct_indexes + missing_indexes
            ids = [x for index, x in enumerate(ids) if index in correct_indexes]
            X = X[correct_indexes]
            y = [x for index, x in enumerate(y) if index in correct_indexes]

            #train_images_indexes = [x for index, x in enumerate(train_images_indexes) if index in correct_indexes]
            current_it += 1

        return ids, X, y


