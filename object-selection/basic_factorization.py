import numpy as np
import random
import matplotlib.pyplot as plt
from skfusion import fusion
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set()
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, label_ranking_loss
from os import path
import copy
from preselection import Preselection


SELECTION_ALGORITHM_VALUES = ['knn', 'rf', 'random']
EVALUATION_METRIC_VALUES = ['auc', 'rmse', 'lrl']


def rmse(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2) / y_true.size)


class BasicFactorization:
    def __init__(self, selection_algorithm='knn', show_selection_results=False, test_set_ratio=0.2):
        if selection_algorithm not in SELECTION_ALGORITHM_VALUES:
            print('Error: wrong selection algorithm')
            return

        self.selection_algorithm = selection_algorithm
        self.test_set_ratio = test_set_ratio
        self.show_selection_results = show_selection_results
        self.ids = []
        self.features = []
        self.ratings = []
        self.users_ratings = []
        self.users = []
        self.objects_for_ids = {}
        self.unique_ratings = []
        self.unique_ids = []
        self.ids_to_i = {}
        self.selected_features = []
        self.predictions = []
        self.mask = []
        self.true_values = []
        self.t = 6
        self.z_score = False
        self.mns = 0
        self.sstd = 0

    def save_objects_for_ids(self):
        """
        Saves indexes of objects for each id to self.objects_for_ids
        """
        for id_ in self.unique_ids:
            self.objects_for_ids[id_] = []

        for i, id_ in enumerate(self.ids):
            self.objects_for_ids[id_].append(i)

    def split_object_selection_data(self, current_id):
        """
        Divides features and ratings into train and test set. Test set are all objects with current_id,
        train set are all other objects.

        :param current_id: Id for objects for test set
        :returns:  train X, test X, train y,  test y
        """

        test_indexes = []
        train_indexes = []

        for i, id_ in enumerate(self.ids):
            if id_ == current_id:
                test_indexes.append(i)
            else:
                train_indexes.append(i)

        train_X = self.features[train_indexes, :]
        test_X = self.features[test_indexes, :]
        train_y = [self.ratings[j] for j in train_indexes]
        test_y = [self.ratings[j] for j in test_indexes]

        return train_X, test_X, train_y, test_y, test_indexes

    def show_object_selection_results(self, predicted_ratings, true_ratings):
        """
        Prints classification predictions for object selection (classification accuracy and confusion matirx)

        :param predicted_ratings: Vector of predicted ratings
        :param true_ratings: Vector of true ratings
        """
        print('\nDraw object selection results')
        ca = accuracy_score(true_ratings, predicted_ratings)
        print("\nClassification Accuracy")
        print(ca)
        print()

        cm = confusion_matrix(true_ratings, predicted_ratings, labels=self.unique_ratings)
        df_cm = pd.DataFrame(cm, index=self.unique_ratings,
                             columns=self.unique_ratings)
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, annot=True, cmap='YlGnBu', fmt='d', annot_kws={"size": 22})
        plt.title('Kontingenčna matrika', fontsize=16)
        plt.xlabel('Napovedane vrednosti', fontsize=14)
        plt.ylabel('Pravilne vrednosti', fontsize=14)
        plt.show()

    def get_random_objects(self):
        """
        Selects a random object for each id

        :returns: selected_objects_i_for_ids - object with ids as keys and indexes of objects as values
        """
        selected_objects_i_for_ids = {}
        for id_ in self.unique_ids:
            current_objects = self.objects_for_ids[id_]
            n = len(current_objects)
            selected = random.randint(0, n-1)
            selected_objects_i_for_ids[id_] = current_objects[selected]
        return selected_objects_i_for_ids

    def object_selection(self):
        """
        Selects objects for each id with algorithm specified in self.selected_algorithm

        :returns: selected_objects_i_for_ids - object with ids as keys and indexes of objects as values
        """
        print('Object selection: ' + self.selection_algorithm)

        if self.selection_algorithm == 'rf':
            model = RandomForestClassifier()
        elif self.selection_algorithm == 'knn':
            model = KNeighborsClassifier(n_neighbors=3)
        else:
            return self.get_random_objects()

        true_ratings = {}
        predicted_ratings = {}
        selected_objects_i_for_ids = {}

        for current_id in self.unique_ids:
            train_X, test_X, train_y, test_y, test_indexes = self.split_object_selection_data(current_id)

            if len(test_X) == 0:
                continue

            model.fit(train_X, train_y)
            predicted_categories_probabilities = model.predict_proba(test_X)
            ordered_classes = list(set(train_y.copy()))
            ordered_classes.sort()
            predicted_categories = []
            best_probabilities = []

            for p in predicted_categories_probabilities:
                max_probability = max(p)
                max_index = list(p).index(max_probability)
                selected_class = ordered_classes[max_index]
                predicted_categories.append(selected_class)
                best_probabilities.append(max_probability)

            # Save predictions to object
            predicted_ratings[current_id] = predicted_categories
            true_ratings[current_id] = test_y

            # Save one object with the most accurate prediction
            errors = [abs(predicted_categories[i] - test_y[i]) for i in range(len(test_y))]
            min_error = min(errors)

            # Find objects with min errors
            min_errors_indexes = []
            for index, error in enumerate(errors):
                if error == min_error:
                    min_errors_indexes.append(index)

            # Find object with best probability in min error array
            selected_probabilities = [p for index, p in enumerate(best_probabilities) if index in min_errors_indexes]
            max_selected_probability = max(selected_probabilities)
            max_selected_index = selected_probabilities.index(max_selected_probability)
            selected_objects_i_for_ids[current_id] = min_errors_indexes[max_selected_index]

        predicted_ratings_vector = []
        ratings_vector = []
        for index, key in enumerate(true_ratings.keys()):
            for el in predicted_ratings[key]:
                predicted_ratings_vector.append(el)
            for el in true_ratings[key]:
                ratings_vector.append(el)

        if self.show_selection_results:
            self.show_object_selection_results(predicted_ratings_vector, ratings_vector)

        return selected_objects_i_for_ids

    def save_ids_to_i(self):
        """
        Saves transformation from ids to indexes (0 to n_ids-1) in data matrices to self.ids_to_i
        """
        ids_to_i = {}
        next_id = 0
        for id_ in self.unique_ids:
            ids_to_i[id_] = next_id
            next_id += 1
        self.ids_to_i = ids_to_i

    def save_data_for_factorization(self, selected_objects_i_for_ids):
        """
        Saves selected objects to self.selected_features

        :param selected_objects_i_for_ids: selected object for each provider
        """
        n_objects = len(selected_objects_i_for_ids.keys())
        selected_features = np.zeros((n_objects, self.features.shape[1]))
        for key in selected_objects_i_for_ids.keys():
            index = self.ids_to_i[key]
            selected_features[index, :] = self.features[selected_objects_i_for_ids[key], :]
        self.selected_features = selected_features

    def split_train_test(self, X, ratio):
        """
        Splits data to train and test set (1 - ratio of each user's ratings to train set and ratio ratings to test set)

        :param X: ratings matrix
        :param ratio: ratio of ratings for each user for test set

        :returns: mask (mask[i, j] = 1 for test set)
        """
        mask = np.full(X.shape, False)

        # For each provider select ratings for test set
        for j in range(X.shape[1]):
            indexes = []
            # Find indexes where rating != 0
            current_provider = X[:, j]
            for index, r in enumerate(current_provider):
                if r != 0:
                    indexes.append(index)

            selected_indexes = []
            if len(indexes) >= 1 / ratio:
                selected_indexes = random.choices(indexes, k=round(len(indexes) * ratio))
            elif len(indexes) > 1:
                selected_indexes = [random.choice(indexes)]
            for i in selected_indexes:
                mask[i][j] = True

        return mask

    def get_cv_masks(self, X, mask, k):
        """
        Makes k masks for cv

        :param X: ratings matrix
        :param mask: mask for primary test and train set
        :param k: number of cv scores for each parameters combination

        :returns: cv_masks (array of k masks)
        """
        cv_masks = []

        # Initialize new masks for each cv iteration
        for _ in range(k):
            new_mask = copy.deepcopy(mask)
            cv_masks.append(new_mask)

        # Fill masks for each provider
        for j in range(X.shape[1]):
            indexes = []
            # Find indexes where rating != 0 and is not in the test set
            current_provider = X[:, j]
            for index, r in enumerate(current_provider):
                if r != 0 and mask[index][j] != 1:
                    indexes.append(index)
            indexes = np.array(indexes)
            random.shuffle(indexes)
            split_indexes = np.array_split(indexes, k)

            # Fill mask for each k (cv iteration)
            for mask_index, k_indexes in enumerate(split_indexes):
                for i in k_indexes:
                    cv_masks[mask_index][i][j] = True

        return cv_masks

    def load_results(self, results_file):
        """
        Load results

        :param results_file: path to previously computed scores for cv
        """

        data = pd.read_csv(results_file, sep=',',
                           names=['p_t1', 'p_t2', 'p_t3', 'p_t4', 'score'])
        all_p_t1 = list(data['p_t1'])[1:]
        all_p_t2 = list(data['p_t2'])[1:]
        all_p_t3 = list(data['p_t3'])[1:]
        all_p_t4 = list(data['p_t4'])[1:]
        all_scores = list(data['score'])[1:]

        all_scores_np = np.array(all_scores)
        index_max = np.argmin(all_scores_np)

        return all_p_t1[index_max], all_p_t2[index_max], all_p_t3[index_max], all_p_t4[index_max]

    def cross_validation(self, k, parameters, parameters_t, mask, R12, results_file):
        """
        Makes k masks for cv

        :param k: number of cv masks for each parameter combination
        :param parameters: array of parameters for cross validation
        :param mask: mask for primary test and train set
        :param R12: matrix for dfmf
        :param R23: matrix for dfmf
        :param R14: matrix for dfmf
        :param results_file: file for saving cv scores

        :returns: best_p_t1, best_p_t2, best_p_t3, best_p_t4 (best parameters)
        """
        if path.exists(results_file):
            return self.load_results(results_file)

        cv_masks = self.get_cv_masks(self.users_ratings, mask, k)

        #best_cv_score = math.inf
        best_cv_score = 0
        best_p_t1 = 0
        best_p_t2 = 0
        best_t = 0

        all_p_t1 = []
        all_p_t2 = []
        all_t = []
        all_scores = []
        original_R12 = R12.copy()
        for p_t1 in parameters:
            for p_t2 in parameters:
                for t in parameters_t:
                    scores = []
                    for current_cv_mask in cv_masks:
                        t1 = fusion.ObjectType('Type 1', p_t1)
                        t2 = fusion.ObjectType('Type 2', p_t2)

                        R12 = original_R12.copy()

                        for i in range(current_cv_mask.shape[0]):
                            for j in range(current_cv_mask.shape[1]):
                                if current_cv_mask[i][j] or original_R12[i, j] == 0:
                                    R12[i][j] = np.NaN

                        mns = 0
                        sstd = 0
                        if self.z_score:
                            mns = np.nanmean(a=R12, axis=0, keepdims=True)
                            sstd = np.nanstd(a=R12, axis=0, keepdims=True)
                            R12 = (R12 - mns) / sstd

                        relations = [fusion.Relation(R12, t1, t2, name='Ratings')]
                        fusion_graph = fusion.FusionGraph()
                        fusion_graph.add_relations_from(relations)

                        fuser = fusion.Dfmf(init_type="random_vcol")
                        #fusion_graph['Ratings'].mask = current_cv_mask
                        dfmf_mod = fuser.fuse(fusion_graph)

                        R12_pred = dfmf_mod.complete(fusion_graph['Ratings'])

                        predictions = R12_pred
                        mask = current_cv_mask
                        true_values = original_R12

                        if self.z_score:
                            predictions = (predictions * sstd) + mns

                        ratings_true = []
                        ratings_predicted = []

                        for i in range(predictions.shape[0]):
                            for j in range(predictions.shape[1]):
                                if mask[i][j]:
                                    ratings_true.append(true_values[i][j])
                                    ratings_predicted.append(predictions[i][j])

                        new_ratings_true = []
                        new_ratings_predicted = []
                        for r_true, r_predicted in zip(ratings_true, ratings_predicted):
                            if r_true > t:
                                new_ratings_true.append(2)
                            else:
                                new_ratings_true.append(1)
                            if r_predicted > t:
                                new_ratings_predicted.append(2)
                            else:
                                new_ratings_predicted.append(1)
                        ratings_true = new_ratings_true
                        ratings_predicted = new_ratings_predicted


                        ratings_true = np.asarray(ratings_true)
                        ratings_predicted = np.asarray(ratings_predicted)

                        # Rmse
                        score = roc_auc_score(ratings_true, ratings_predicted)
                        print(score)
                        #score = rmse(ratings_true, ratings_predicted)
                        #print('\nrmse: ' + str(score))
                        scores.append(score)

                    score = sum(scores) / len(scores)
                    all_p_t1.append(p_t1)
                    all_p_t2.append(p_t2)
                    all_t.append(t)
                    all_scores.append(score)

                    # Save best scores to a variable

                    if score >= best_cv_score:
                        best_cv_score = score
                        best_p_t1 = p_t1
                        best_p_t2 = p_t2
                        best_t = t

        # Save cv scores to a csv file
        data = {'p_t1': all_p_t1,
                'p_t2': all_p_t2,
                't': all_t,
                'score': all_scores}
        df = pd.DataFrame(data, columns=['p_t1', 'p_t2', 't', 'score'])
        df.to_csv(results_file)

        return best_p_t1, best_p_t2, best_t

    def factorization(self, cv_results_file):
        """
        Matrix factorization, saves predictions to self.predictions and mask to self.mask

        :param cv_results_file: file for saving cv scores
        """
        print('\nDfmf')
        mask = self.split_train_test(self.users_ratings, 0.2)

        R12 = self.users_ratings

        # best_p_t1, best_p_t2, best_p_t3, best_p_t4
        best_p_t1 = 100
        best_p_t2 = 100

        t = [6, 7, 8]
        parameters = [2, 4, 6, 8, 10, 50, 75, 100, 125, 150]
        k = 3
        #best_p_t1, best_p_t2, best_p_t3, best_p_t4, t = 70, 70, 8, 10, 6
        best_p_t1, best_p_t2, best_t = self.cross_validation(k, parameters, t, mask, R12, cv_results_file)
        print(str(best_p_t1) + ' ' + str(best_p_t2) + ' ' + str(best_t) + '\n')
        self.t = best_t

        new_R12 = np.zeros(R12.shape)
        for i in range(R12.shape[0]):
            for j in range(R12.shape[1]):
                if R12[i][j] == 0:
                    new_R12[i][j] = np.NaN
                else:
                    new_R12[i][j] = R12[i][j]
        R12 = new_R12.copy()
        #R12 = new_R12

        if self.z_score:
            mns = np.nanmean(a=R12, axis=0, keepdims=True)
            sstd = np.nanstd(a=R12, axis=0, keepdims=True)
            R12 = (R12 - mns) / sstd
            self.mns = mns
            self.sstd = sstd

        # Predictions
        t1 = fusion.ObjectType('Type 1', best_p_t1)
        t2 = fusion.ObjectType('Type 2', best_p_t2)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j]:
                    R12[i][j] = np.NaN

        relations = [fusion.Relation(R12, t1, t2, name='Ratings')]
        fusion_graph = fusion.FusionGraph()
        fusion_graph.add_relations_from(relations)

        fuser = fusion.Dfmf(init_type="random_vcol")
        #fusion_graph['Ratings'].mask = mask.astype('bool')
        dfmf_mod = fuser.fuse(fusion_graph)

        R12_pred = dfmf_mod.complete(fusion_graph['Ratings'])

        self.predictions = R12_pred
        self.mask = mask
        self.true_values = new_R12

    def transform(self, ids, features, ratings, users_ratings, users, cv_results_file, images_indexes, true_objects_indexes, false_objects_indexes, paths, z_score=False):
        """
        Calculates latent matrices and saves ratings predictions

        :param ids: vector of ids, rows indexes corresponding to features and ratings indexes
        :param features: matrix of features/objects as rows
        :param ratings: vector of average ratings
        :param users_ratings: matrix of all users ratings for experiences, rows indexes corresponding to users indexes
        :param users: matrix of additional data for users
        :param cv_results_file: ile for saving cv scores
        """
        if self.selection_algorithm != 'random':
            preselection = Preselection(true_objects_indexes, false_objects_indexes)
            ids, features, ratings = preselection.transform(paths, ids, ratings, features)

        self.ids = ids
        self.features = features
        self.ratings = ratings
        self.users_ratings = users_ratings
        self.users = users
        self.unique_ratings = list(set(ratings))
        self.unique_ids = list(set(self.ids))
        self.z_score = z_score

        self.save_objects_for_ids()
        self.save_ids_to_i()

        selected_objects_i_for_ids = self.object_selection()
        self.save_data_for_factorization(selected_objects_i_for_ids)
        self.factorization(cv_results_file)

    def evaluate(self, evaluation_metric='auc'):
        """
        Prints evaluation score

        :param evaluation_metric: string name of the evaluation metric specified in EVALUATION_METRIC_VALUES
        """

        if evaluation_metric not in EVALUATION_METRIC_VALUES:
            print('Error: wrong evaluation metric')
            return

        predictions = self.predictions
        mask = self.mask
        true_values = self.true_values

        if self.z_score:
            a = np.asanyarray(predictions)
            mns = self.mns
            sstd = self.sstd
            predictions = (a * sstd) + mns

        ratings_true = []
        ratings_predicted = []

        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                if mask[i][j]:
                    ratings_true.append(true_values[i, j])
                    ratings_predicted.append(predictions[i, j])

        if True:
            new_ratings_true = []
            new_ratings_predicted = []
            for r_true, r_predicted in zip(ratings_true, ratings_predicted):
                if r_true > self.t:
                    new_ratings_true.append(2)
                else:
                    new_ratings_true.append(1)
                if r_predicted > self.t:
                    new_ratings_predicted.append(2)
                else:
                    new_ratings_predicted.append(1)
            ratings_true = new_ratings_true
            ratings_predicted = new_ratings_predicted

        ratings_true = np.asarray(ratings_true)
        ratings_predicted = np.asarray(ratings_predicted)

        ratings_true = np.asarray(ratings_true)
        ratings_predicted = np.asarray(ratings_predicted)

        # Rmse
        score_rmse = rmse(ratings_true, ratings_predicted)
        print('\nrmse: ' + str(score_rmse))
        # Auc
        score_auc = roc_auc_score(ratings_true, ratings_predicted)
        print('\nauc: ' + str(score_auc))
        return score_auc, score_rmse


