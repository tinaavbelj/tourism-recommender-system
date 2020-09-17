import seaborn as sns; sns.set()
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import pandas as pd
from sklearn.metrics import accuracy_score
from statistics import median
from os import path
import random
from sklearn.model_selection import KFold
import copy
from matplotlib import rcParams

from utils import get_features


E = 0.005
MAX_IT = 10


class Preselection:
    def __init__(self, data_directory, features_path, true_objects_indexes, false_objects_indexes):
        self.data_directory = data_directory
        self.features_path = features_path
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
        categories_vector = [name.split('-')[1] for name in file_names]
        ratings_vector = [int(name.split('-')[-2]) for name in file_names]
        features = get_features(self.features_path, images_paths)
        images_indexes = [name.split('-')[3].split('.')[0] for name in file_names]

        # Split data
        train_X, test_X, train_y, test_y, train_ids, test_ids, train_indexes, test_indexes = self.split_providers(ids_vector, ratings_vector, features)

        current_it = 0
        X = train_X
        y = train_y
        train_images_indexes = [x for index, x in enumerate(images_indexes) if index in train_indexes]

        # Test on all
        model = KNeighborsClassifier()
        model.fit(X, y)
        predicted = model.predict(test_X)
        ca = accuracy_score(test_y, predicted)

        print('- - - - -')
        print(X.shape)
        print('CA')
        print(ca)
        print('- - - - -')

        true_detected = 0
        false_detected = 0
        false_true_detected = 0
        false_false_detected = 0
        other = 0
        for i, index in enumerate(test_indexes):
            p = predicted[i] == test_y[i]
            if images_indexes[index] in str(self.true_objects_indexes) and p:
                true_detected += 1
            elif images_indexes[index] in str(self.false_objects_indexes) and not p:
                false_detected += 1
            elif images_indexes[index] in str(self.false_objects_indexes) and p:
                false_true_detected += 1
            elif images_indexes[index] in str(self.true_objects_indexes) and not p:
                false_false_detected += 1
            else:
                other += 1

        print('TP: ' + str(true_detected))
        print('TN: ' + str(false_detected))
        print('FP: ' + str(false_true_detected))
        print('FN: ' + str(false_false_detected))
        print('- - - - -\n')

        print(X.shape)
        print(len(y))
        print()
        ca = 0
        ca_new = 1

        # draw
        data = copy.deepcopy(train_images_indexes)
        data.sort()
        n_bins = list(set(images_indexes)).sort()
        print(n_bins)

        print()
        print('N TRUE: ')
        print(len([x for x in train_images_indexes if int(x) in self.true_objects_indexes]))
        print('N FALSE: ')
        print(len([x for x in train_images_indexes if int(x) in self.false_objects_indexes]))
        print()

        SMALL_SIZE = 16
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 22

        rcParams.update({'figure.autolayout': True})
        plt.tight_layout()
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.xlabel('Indeks slike')
        plt.ylabel('Število slik')
        plt.title('Pred izbiranjem')
        plt.ylim(0, 250)

        _, _, patches = plt.hist(data, bins=n_bins)
        # plt.show()

        # fig, ax = plt.subplots()
        # data = train_images_indexes

        # N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1)

        print(len(self.true_objects_indexes))
        print(len(self.false_objects_indexes))
        for i in range(0, len(self.true_objects_indexes)):
            patches[i].set_facecolor('b')
        for i in range(len(self.true_objects_indexes), 10):
            patches[i].set_facecolor('r')

        plt.show()

        while current_it < MAX_IT and abs(ca - ca_new) > E:
            correct_indexes = []
            kf = KFold(n_splits=3)
            ca = ca_new
            kfolds_ca = []
            print('- \n')
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
                for index, p in enumerate(predicted):
                    pass
                    #print(p == current_test_y[index])
                    #print(current_images_indexes[index])
                    #print('-')
                correct_indexes = correct_indexes + [test_index[index] for index, p in enumerate(predicted) if p == current_test_y[index]]
            ca_new = sum(kfolds_ca) / len(kfolds_ca)
            print('CA')
            print(ca_new)
            print()

            X = X[correct_indexes]
            y = [x for index, x in enumerate(y) if index in correct_indexes]
            train_images_indexes = [x for index, x in enumerate(train_images_indexes) if index in correct_indexes]
            current_it += 1
            print('Shape')
            print(X.shape)
            print(len(y))
            print()
            print('- \n')

            data = copy.deepcopy(train_images_indexes)
            data = [int(x) for x in data]
            data.sort()
            n_bins = list(set(images_indexes)).sort()
            print(n_bins)

            print()
            print('N TRUE: ')
            print(len([x for x in train_images_indexes if int(x) in self.true_objects_indexes]))
            print('N FALSE: ')
            print(len([x for x in train_images_indexes if int(x) in self.false_objects_indexes]))
            print()

            SMALL_SIZE = 16
            MEDIUM_SIZE = 16
            BIGGER_SIZE = 22

            rcParams.update({'figure.autolayout': True})
            plt.tight_layout()
            plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            plt.xlabel('Indeks slike')
            plt.ylabel('Število slik')
            plt.title(str(current_it) + '. iteracija')
            plt.ylim(0, 250)

            _, _, patches = plt.hist(data, bins=n_bins)
            #plt.show()

            #fig, ax = plt.subplots()
            #data = train_images_indexes

            #N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1)

            for i in range(0, len(self.true_objects_indexes)):
                patches[i].set_facecolor('b')
            for i in range(len(self.true_objects_indexes), 10):
                patches[i].set_facecolor('r')

            plt.show()


        train_images_indexes.sort()
        n_bins = list(set(images_indexes)).sort()
        plt.hist(train_images_indexes, bins=n_bins)
        plt.show()

        fig, ax = plt.subplots()
        data = train_images_indexes

        print()
        print('N TRUE: ')
        print(len([x for x in train_images_indexes if int(x) in self.true_objects_indexes]))
        print('N FALSE: ')
        print(len([x for x in train_images_indexes if int(x) in self.false_objects_indexes]))
        print()

        SMALL_SIZE = 16
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 22

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.xlabel('Indeks slike')
        plt.ylabel('Število slik')
        plt.title(str(current_it + 1) + '. iteracija')
        plt.ylim(0, 250)

        N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1)

        print(len(self.true_objects_indexes))
        print(len(self.false_objects_indexes))
        for i in range(0, len(self.true_objects_indexes)):
            patches[i].set_facecolor('b')
        for i in range(len(self.true_objects_indexes), 10):
            patches[i].set_facecolor('r')

        plt.show()

        model = KNeighborsClassifier()
        model.fit(X, y)
        predicted = model.predict(test_X)
        ca = accuracy_score(test_y, predicted)

        print('- - - - -')
        print(X.shape)
        print('CA')
        print(ca)
        print('- - - - -')

        true_detected = 0
        false_detected = 0
        false_true_detected = 0
        false_false_detected = 0
        other = 0
        for i, index in enumerate(test_indexes):
            p = predicted[i] == test_y[i]
            if images_indexes[index] in str(self.true_objects_indexes) and p:
                true_detected += 1
            elif images_indexes[index] in str(self.false_objects_indexes) and not p:
                false_detected += 1
            elif images_indexes[index] in str(self.false_objects_indexes) and p:
                false_true_detected += 1
            elif images_indexes[index] in str(self.true_objects_indexes) and not p:
                false_false_detected += 1
            else:
                other += 1

        print('TP: ' + str(true_detected))
        print('TN: ' + str(false_detected))
        print('FP: ' + str(false_true_detected))
        print('FN: ' + str(false_false_detected))
        print('- - - - -\n')

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



