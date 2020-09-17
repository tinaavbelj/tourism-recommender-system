import seaborn as sns; sns.set()
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams


def main():
    # images 8-2

    results_file = './scores/generated-data-r-2-n-8-2-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random = list(data['scores'])[1:]
    scores_random = [float(x) for x in scores_random]

    results_file = './scores/generated-data-r-2-n-8-2-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn = list(data['scores'])[1:]
    scores_knn = [float(x) for x in scores_knn]

    results_file = './scores/generated-data-r-2-n-8-2-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf = list(data['scores'])[1:]
    scores_rf = [float(x) for x in scores_rf]

    results_file = './scores/generated-data-r-2-n-8-2-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b = list(data['scores'])[1:]
    scores_b = [float(x) for x in scores_b]

    results_file = './scores/generated-data-nr-2-n-8-2-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn_centered = list(data['scores'])[1:]
    scores_knn_centered = [float(x) for x in scores_knn_centered]

    results_file = './scores/generated-data-nr-2-n-8-2-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random_centered = list(data['scores'])[1:]
    scores_random_centered = [float(x) for x in scores_random_centered]

    results_file = './scores/generated-data-nr-2-n-8-2-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf_centered = list(data['scores'])[1:]
    scores_rf_centered = [float(x) for x in scores_rf_centered]

    results_file = './scores/generated-data-nr-2-n-8-2-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b_centered = list(data['scores'])[1:]
    scores_b_centered = [float(x) for x in scores_b_centered]

    label = ['knn', 'random', 'random forest', 'knn centered', 'random centered', 'rf centered', 'b centered']
    plt.figure()
    plt.title('AUC za različne načine izbiranja virov')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')
    ax = plt.subplot(111)
    pos = [1, 2, 3, 4, 6, 7, 8, 9]
    plt.violinplot([scores_knn, scores_random, scores_rf, scores_b, scores_knn_centered, scores_random_centered, scores_rf_centered, scores_b_centered], pos, vert=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    plt.show()

    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    rcParams.update({'figure.autolayout': True})
    plt.figure()
    plt.title('Slike: 80% pravilnih, 20% napačnih')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')

    ax = plt.subplot(111)
    label = ['Brez zlivanja', 'Naključno', 'KNN', 'RF', 'Brez zlivanja', 'Naključno', 'KNN', 'RF']

    ax.set_xticks(pos)
    ax.set_xticklabels(label, rotation=90)
    positions = [1, 2, 3, 4]
    add_label(ax.violinplot([scores_b, scores_random, scores_knn, scores_rf], positions), "Necentrirani podatki")

    positions = [6, 7, 8, 9]

    add_label(ax.violinplot([scores_b_centered, scores_random_centered, scores_knn_centered, scores_rf_centered], positions), "Centrirani podatki")
    ax.legend(*zip(*labels), loc=3)
    plt.show()

    # images 6-4

    results_file = './scores/generated-data-r-2-n-6-4-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random = list(data['scores'])[1:]
    scores_random = [float(x) for x in scores_random]

    results_file = './scores/generated-data-r-2-n-6-4-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn = list(data['scores'])[1:]
    scores_knn = [float(x) for x in scores_knn]

    results_file = './scores/generated-data-r-2-n-6-4-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf = list(data['scores'])[1:]
    scores_rf = [float(x) for x in scores_rf]

    results_file = './scores/generated-data-r-2-n-6-4-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b = list(data['scores'])[1:]
    scores_b = [float(x) for x in scores_b]

    results_file = './scores/generated-data-nr-2-n-6-4-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn_centered = list(data['scores'])[1:]
    scores_knn_centered = [float(x) for x in scores_knn_centered]

    results_file = './scores/generated-data-nr-2-n-6-4-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random_centered = list(data['scores'])[1:]
    scores_random_centered = [float(x) for x in scores_random_centered]

    results_file = './scores/generated-data-nr-2-n-6-4-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf_centered = list(data['scores'])[1:]
    scores_rf_centered = [float(x) for x in scores_rf_centered]

    results_file = './scores/generated-data-nr-2-n-6-4-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b_centered = list(data['scores'])[1:]
    scores_b_centered = [float(x) for x in scores_b_centered]

    label = ['knn', 'random', 'random forest', 'knn centered', 'random centered', 'rf centered', 'b centered']
    plt.figure()
    plt.title('AUC za različne načine izbiranja virov (6-4)')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')
    ax = plt.subplot(111)
    pos = [1, 2, 3, 4, 6, 7, 8, 9]
    plt.violinplot([scores_knn, scores_random, scores_rf, scores_b, scores_knn_centered, scores_random_centered, scores_rf_centered, scores_b_centered], pos, vert=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    plt.show()


    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    rcParams.update({'figure.autolayout': True})
    plt.figure()
    plt.title('Slike: 60% pravilnih, 40% napačnih')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')

    ax = plt.subplot(111)
    label = ['Brez zlivanja', 'Naključno', 'KNN', 'RF', 'Brez zlivanja', 'Naključno', 'KNN', 'RF']

    ax.set_xticks(pos)
    ax.set_xticklabels(label, rotation=90)
    positions = [1, 2, 3, 4]
    add_label(ax.violinplot([scores_b, scores_random, scores_knn, scores_rf], positions), "Necentrirani podatki")

    positions = [6, 7, 8, 9]

    add_label(ax.violinplot([scores_b_centered, scores_random_centered, scores_knn_centered, scores_rf_centered], positions), "Centrirani podatki")
    ax.legend(*zip(*labels), loc=3)
    plt.show()


    # text 02

    results_file = './scores/generated-data-r-2-n-02-l-100-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random = list(data['scores'])[1:]
    scores_random = [float(x) for x in scores_random]

    results_file = './scores/generated-data-r-2-n-02-l-100-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn = list(data['scores'])[1:]
    scores_knn = [float(x) for x in scores_knn]

    results_file = './scores/generated-data-r-2-n-02-l-100-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf = list(data['scores'])[1:]
    scores_rf = [float(x) for x in scores_rf]

    results_file = './scores/generated-data-r-2-n-02-l-100-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b = list(data['scores'])[1:]
    scores_b = [float(x) for x in scores_b]

    results_file = './scores/generated-data-nr-2-n-02-l-100-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn_centered = list(data['scores'])[1:]
    scores_knn_centered = [float(x) for x in scores_knn_centered]

    results_file = './scores/generated-data-nr-2-n-02-l-100-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random_centered = list(data['scores'])[1:]
    scores_random_centered = [float(x) for x in scores_random_centered]

    results_file = './scores/generated-data-nr-2-n-02-l-100-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf_centered = list(data['scores'])[1:]
    scores_rf_centered = [float(x) for x in scores_rf_centered]

    results_file = './scores/generated-data-nr-2-n-02-l-100-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b_centered = list(data['scores'])[1:]
    scores_b_centered = [float(x) for x in scores_b_centered]

    label = ['knn', 'random', 'random forest', 'knn centered', 'random centered', 'rf centered', 'b centered']
    plt.figure()
    plt.title('AUC za različne načine izbiranja virov, besedila 02')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')
    ax = plt.subplot(111)
    pos = [1, 2, 3, 4, 6, 7, 8, 9]
    plt.violinplot([scores_knn, scores_random, scores_rf, scores_b, scores_knn_centered, scores_random_centered,
                    scores_rf_centered, scores_b_centered], pos, vert=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    plt.show()


    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    rcParams.update({'figure.autolayout': True})
    plt.figure()
    plt.title('Besedila: 80% pravilnih, 20% napačnih')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')

    ax = plt.subplot(111)
    label = ['Brez zlivanja', 'Naključno', 'KNN', 'RF', 'Brez zlivanja', 'Naključno', 'KNN', 'RF']

    ax.set_xticks(pos)
    ax.set_xticklabels(label, rotation=90)
    positions = [1, 2, 3, 4]
    add_label(ax.violinplot([scores_b, scores_random, scores_knn, scores_rf], positions), "Necentrirani podatki")

    positions = [6, 7, 8, 9]

    add_label(ax.violinplot([scores_b_centered, scores_random_centered, scores_knn_centered, scores_rf_centered], positions), "Centrirani podatki")
    ax.legend(*zip(*labels), loc=3)
    plt.show()

    # text 04

    results_file = './scores/generated-data-r-2-n-04-l-100-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random = list(data['scores'])[1:]
    scores_random = [float(x) for x in scores_random]

    results_file = './scores/generated-data-r-2-n-04-l-100-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn = list(data['scores'])[1:]
    scores_knn = [float(x) for x in scores_knn]

    results_file = './scores/generated-data-r-2-n-04-l-100-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf = list(data['scores'])[1:]
    scores_rf = [float(x) for x in scores_rf]

    results_file = './scores/generated-data-r-2-n-04-l-100-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b = list(data['scores'])[1:]
    scores_b = [float(x) for x in scores_b]

    results_file = './scores/generated-data-nr-2-n-04-l-100-knn.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_knn_centered = list(data['scores'])[1:]
    scores_knn_centered = [float(x) for x in scores_knn_centered]

    results_file = './scores/generated-data-nr-2-n-04-l-100-random.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_random_centered = list(data['scores'])[1:]
    scores_random_centered = [float(x) for x in scores_random_centered]

    results_file = './scores/generated-data-nr-2-n-04-l-100-rf.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_rf_centered = list(data['scores'])[1:]
    scores_rf_centered = [float(x) for x in scores_rf_centered]

    results_file = './scores/generated-data-nr-2-n-04-l-100-b.csv'
    data = pd.read_csv(results_file, sep=',', names=['scores'])
    scores_b_centered = list(data['scores'])[1:]
    scores_b_centered = [float(x) for x in scores_b_centered]

    label = ['knn', 'random', 'random forest', 'knn centered', 'random centered', 'rf centered', 'b centered']
    plt.figure()
    plt.title('AUC za različne načine izbiranja virov, besedila 04')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')
    ax = plt.subplot(111)
    pos = [1, 2, 3, 4, 6, 7, 8, 9]
    plt.violinplot([scores_knn, scores_random, scores_rf, scores_b, scores_knn_centered, scores_random_centered,
                    scores_rf_centered, scores_b_centered], pos, vert=True)
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    plt.show()


    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    rcParams.update({'figure.autolayout': True})
    plt.figure()
    plt.title('Besedila: 60% pravilnih, 40% napačnih')
    plt.ylabel('AUC')
    plt.xlabel('Algoritem izbiranja')

    ax = plt.subplot(111)
    label = ['Brez zlivanja', 'Naključno', 'KNN', 'RF', 'Brez zlivanja', 'Naključno', 'KNN', 'RF']

    ax.set_xticks(pos)
    ax.set_xticklabels(label, rotation=90)
    positions = [1, 2, 3, 4]
    add_label(ax.violinplot([scores_b, scores_random, scores_knn, scores_rf], positions), "Necentrirani podatki")

    positions = [6, 7, 8, 9]

    add_label(ax.violinplot([scores_b_centered, scores_random_centered, scores_knn_centered, scores_rf_centered], positions), "Centrirani podatki")
    ax.legend(*zip(*labels), loc=3)
    plt.show()

if __name__ == '__main__':
    main()
