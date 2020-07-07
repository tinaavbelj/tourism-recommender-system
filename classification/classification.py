import seaborn as sns; sns.set()
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from statistics import median
from sklearn import cluster
from os import path
from sklearn import linear_model

from utils import get_features


ALGORITHM_VALUES = ['knn', 'rf', 'lr']


class Classification:
    def __init__(self, data_directory, features_path, algorithm='knn', feature_agglomeration=False):
        if algorithm not in ALGORITHM_VALUES:
            print('Error: wrong algorithm')
            return

        self.algorithm = algorithm
        self.feature_agglomeration = feature_agglomeration
        self.data_directory = data_directory
        self.features_path = features_path

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

    def transform(self, results_file=''):
        """
        Classify images for each provider and save predictions

        :param results_file: path to previously computed predictions
        """
        if path.exists(results_file):
            self.load_results(results_file)
            return

        file_names = os.listdir(self.data_directory)
        images_paths = [self.data_directory + '/' + name for name in file_names]
        ids_vector = [name.split('-')[0] for name in file_names]
        ratings_vector = [int(name.split('-')[-2]) for name in file_names]
        features = get_features(self.features_path, images_paths)

        # Feature Agglomeration
        if self.feature_agglomeration:
            agglomeration = cluster.FeatureAgglomeration(n_clusters=32)
            agglomeration.fit(features)
            features_reduced = agglomeration.transform(features)
            features = features_reduced

        self.unique_ratings = sorted(list(set(ratings_vector)))
        unique_ids = list(set(ids_vector))

        true_ratings_object = {}
        predicted_ratings_object = {}
        predicted_ratings_vector = []
        true_ratings_vector = []
        images_paths_object = {}
        ids_object = {}

        if self.algorithm == 'knn':
            model = KNeighborsClassifier(n_neighbors=3)
        elif self.algorithm == 'lr':
            model = linear_model.Lasso(alpha=0.1)
        else:
            model = RandomForestClassifier()

        for current_id in unique_ids:
            # Images for current_id to test set and other images to train set
            test_indexes = []
            train_indexes = []
            for index, img_id in enumerate(ids_vector):
                if img_id == current_id:
                    test_indexes.append(index)
                else:
                    train_indexes.append(index)

            train_X = features[train_indexes, :]
            test_X = features[test_indexes, :]

            train_y = [ratings_vector[j] for j in train_indexes]
            test_y = [ratings_vector[j] for j in test_indexes]

            if len(test_y) == 0:
                continue

            model.fit(train_X, train_y)
            predictions = model.predict(test_X)

            print(predictions)

            # Save to object
            predicted_ratings_object[current_id] = predictions
            true_ratings_object[current_id] = test_y
            images_paths_object[current_id] = [images_paths[test_index] for test_index in test_indexes]
            ids_object[current_id] = [ids_vector[test_index] for test_index in test_indexes]

            # Save to vector
            predicted_ratings_vector.extend(predictions)
            true_ratings_vector.extend(test_y)

        # Save to class properties
        self.predicted_ratings_object = predicted_ratings_object
        self.true_ratings_object = true_ratings_object
        self.predicted_ratings_vector = predicted_ratings_vector
        self.true_ratings_vector = true_ratings_vector
        self.images_paths_object = images_paths_object
        self.ids_object = ids_object

        # Save predictions to a file
        self.save_results(results_file)

    def violin_plot(self, values_to_plot='average'):
        """
        Violin plot for specified values

        :param values_to_plot: values to plot (average/max/min/median)
        """
        new_ratings_true = []
        new_ratings_predicted = []
        title = ''

        for key in self.true_ratings_object.keys():
            if values_to_plot == 'average':
                current_predicted = [float(i) for i in self.predicted_ratings_object[key]]
                current_true = [float(i) for i in self.true_ratings_object[key]]
                new_ratings_predicted.append(sum(current_predicted) / len(current_predicted))
                new_ratings_true.append(sum(current_true) / len(current_true))
                title = 'Povprečje vseh slik'
            elif values_to_plot == 'max':
                current_predicted = sorted([float(i) for i in self.predicted_ratings_object[key]])[-3:]
                current_true = [float(i) for i in self.true_ratings_object[key]][-3:]
                new_ratings_predicted.append(sum(current_predicted) / len(current_predicted))
                new_ratings_true.append(sum(current_true) / len(current_true))
                title = 'Povprečje najboljših 3 slik'
            elif values_to_plot == 'min':
                current_predicted = sorted([float(i) for i in self.predicted_ratings_object[key]])[:3]
                current_true = [float(i) for i in self.true_ratings_object[key]][:3]
                new_ratings_predicted.append(sum(current_predicted) / len(current_predicted))
                new_ratings_true.append(sum(current_true) / len(current_true))
                title = 'Povprečje najslabših 3 slik'
            elif values_to_plot == 'median':
                current_predicted = [float(i) for i in self.predicted_ratings_object[key]]
                current_true = [float(i) for i in self.true_ratings_object[key]]
                new_ratings_predicted.append(median(current_predicted))
                new_ratings_true.append(median(current_true))
                title = 'Mediana vseh slik'

        ratings_list = []
        for _ in self.unique_ratings:
            ratings_list.append([])

        for index, r in enumerate(new_ratings_true):
            ratings_list[int(r) - 1].append(new_ratings_predicted[index])

        min_plot_value = float(min(self.unique_ratings)) - 0.5
        max_plot_value = float(max(self.unique_ratings)) + 0.5

        plt.title(title)
        plt.ylabel('Napovedana vrednost')
        plt.xlabel('Pravilna vrednost')
        plt.axis((min_plot_value, max_plot_value, min_plot_value, max_plot_value))
        plt.violinplot(ratings_list)
        plt.show()

    def evaluate(self):
        """
        Evaluate classification results with classification accuracy, confusion matrix and violin plots
        """
        # Classification accuracy
        ca = accuracy_score(self.true_ratings_vector, self.predicted_ratings_vector)
        print("\nClassification Accuracy: " + str(ca) + '\n')

        # Confusion matrix
        self.unique_ratings = list(set(self.true_ratings_vector))
        cm = confusion_matrix(self.true_ratings_vector, self.predicted_ratings_vector, labels=self.unique_ratings)
        df_cm = pd.DataFrame(cm, index=self.unique_ratings, columns=self.unique_ratings)
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, cmap='YlGnBu', fmt='d', annot_kws={"size": 22})
        plt.title('Kontingenčna matrika', fontsize=16)
        plt.xlabel('Napovedane vrednosti', fontsize=14)
        plt.ylabel('Pravilne vrednosti', fontsize=14)
        plt.show()

        # Violin plots

        # Average of all images for provider
        self.violin_plot(values_to_plot='average')
        # Average of max 3 images for provider
        self.violin_plot(values_to_plot='max')
        # Average of min 3 images for provider
        self.violin_plot(values_to_plot='min')
        # Median of all images for provider
        self.violin_plot(values_to_plot='median')



