import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import csv
import pickle
import tensorflow as tf
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS

features_path = './data/features-generated-dataset-kranjska-piran'
ratings_path = './data/ratings_kranjska_piran.csv'
data_directory = './data/generated-dataset-kranjska-piran/'

file_names = os.listdir(data_directory)
img_ids_vector = [name.split('-')[0] for name in file_names if name.split('-')[3] == 'image']
categories_vector = [name.split('-')[1] for name in file_names if name.split('-')[3] == 'image']

ratings_object = {}
with open(ratings_path, newline='') as file:
    reader = csv.reader(file, delimiter=';', quotechar='|')
    next(reader, None)
    for row in reader:
        current_id = row[0]
        current_rating = row[3]
        ratings_object[current_id] = current_rating

ratings = []
new_img_ids_vector = []
new_indexes = []
new_img_categories_vector = []

for index, i in enumerate(img_ids_vector):
    if ratings_object.get(i):
        ratings.append(ratings_object[i])
        new_img_ids_vector.append(index)
        new_indexes.append(index)
        new_img_categories_vector.append(categories_vector[index])

categories_vector = new_img_categories_vector
img_ids_vector = new_img_ids_vector

ratings_length = len(ratings_object.keys())
ratings_n = [float(x) for x in ratings]

ratings_new = []
for r in ratings:
    r = float(r)
    if r < 6:
        ratings_new.append(1)
    elif r < 7:
        ratings_new.append(2)
    elif r < 8:
        ratings_new.append(3)
    elif r < 9:
        ratings_new.append(4)
    else:
        ratings_new.append(5)

ratings = ratings_new

with open(features_path, 'rb') as f:
    features = pickle.load(f)
    features = features[new_indexes, :]

    print(len(img_ids_vector))
    print(features.shape)

    colors = {
        'kranjska': 'darkorange',
        'piran': 'tab:blue'}

    #tsne = TSNE(n_components=2)
    #components = tsne.fit_transform(features)

    embedding = MDS(n_components=2)
    components = embedding.fit_transform(features)

    print(components.shape)
    print(len(ratings))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('kranjska-piran images features', fontsize=12)
    ax.set_facecolor('#eaeaf2')
    ax.grid(False)

    for index, point in enumerate(components):
        point_size = 3
        ax.plot(point[0], point[1], marker='o', markersize=point_size, color=colors[categories_vector[index]])
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    ax.legend(custom_lines, colors.keys())
    #plt.show()

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

    title = 'Kategorije'
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_facecolor('#eaeaf2')
    ax.grid(False)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    for index, point in enumerate(components):
        color = colors[categories_vector[index]]
        ax.plot(point[0], point[1], marker='o', markersize=3, color=color)
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    names = ['Kranjska Gora', 'Piran']
    ax.legend(custom_lines, names)
    plt.show()


    #[5.6, 6.0, 6.8, 7.1, 7.2, 7.3, 7.4, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1,
    # 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]
    colors = {
        '7.4': 'r',
        '7.8': 'b',
        '7.9': 'orange',
        '8.0': 'green',
        '8.1': 'yellow',
        '8.2': 'purple',
        '8.3': 'pink',
        '8.4': 'cyan',
        '8.5': 'olive',
        '8.6': 'deeppink',
        '8.7': 'yellowgreen',
        '8.8': 'peachpuff',
        '9.0': 'crimson',
        '9.3': 'tan',
        '9.4': 'lightcoral',
        '9.5': 'slateblue',
        '9.6': 'plum',
        '9.7': 'turquoise',
        '9.8': 'dodgerblue',
        '9.9': 'gold'}

    colors = {
    1: 'crimson',
    2: 'darkorange',
    3: 'gold',
    4: 'tab:blue',
    5: 'darkslateblue'}

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

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Ocene')
    ax.set_facecolor('#eaeaf2')
    ax.grid(False)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    for index, point in enumerate(components):
        point_size = 3
        ax.plot(point[0], point[1], marker='o', markersize=point_size, color=colors[ratings[index]])
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    names = ['[0, 6]', '[6, 7)', '[7, 8)', '[8, 9)', '[9, 10]']
    ax.legend(custom_lines, names)
    plt.show()

    # Ratings Kranjska
    colors = {
    1: 'crimson',
    2: 'darkorange',
    3: 'gold',
    4: 'tab:blue',
    5: 'darkslateblue'}

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

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Ocene Kranjska Gora')
    ax.set_facecolor('#eaeaf2')
    ax.grid(False)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    for index, point in enumerate(components):
        if categories_vector[index] == 'kranjska':
            point_size = 3
            ax.plot(point[0], point[1], marker='o', markersize=point_size, color=colors[ratings[index]])
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    names = ['[0, 6]', '[6, 7)', '[7, 8)', '[8, 9)', '[9, 10]']
    ax.legend(custom_lines, names)
    plt.show()


    # Ratings Piran
    colors = {
    1: 'crimson',
    2: 'darkorange',
    3: 'gold',
    4: 'tab:blue',
    5: 'darkslateblue'}

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

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Ocene Piran')
    ax.set_facecolor('#eaeaf2')
    ax.grid(False)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    for index, point in enumerate(components):
        if categories_vector[index] == 'piran':
            point_size = 3
            ax.plot(point[0], point[1], marker='o', markersize=point_size, color=colors[ratings[index]])
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    names = ['[0, 6]', '[6, 7)', '[7, 8)', '[8, 9)', '[9, 10]']
    ax.legend(custom_lines, names)
    plt.show()


    exit()

    colors = {
        '5.6': 'lightgray',
        '7.4': 'b',
        '7.8': 'lightgray',
        '7.9': 'lightgray',
        '8.0': 'lightgray',
        '8.1': 'lightgray',
        '8.2': 'lightgray',
        '8.3': 'lightgray',
        '8.4': 'lightgray',
        '8.5': 'lightgray',
        '8.6': 'lightgray',
        '8.7': 'lightgray',
        '8.8': 'lightgray',
        '9.0': 'lightgray',
        '9.3': 'lightgray',
        '9.4': 'lightgray',
        '9.5': 'lightgray',
        '9.6': 'lightgray',
        '9.7': 'lightgray',
        '9.8': 'lightgray',
        '9.9': 'r'}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('kranjska-piran images rating', fontsize=12)

    for index, point in enumerate(components):
        point_size = 3
        ax.plot(point[0], point[1], marker='o', markersize=point_size, color=colors[ratings[index]])
    ax.grid()
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    ax.legend(custom_lines, colors.keys())
    plt.show()
    ax.grid(False)

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

    title = 'Kategorije'
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=20)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('kranjska-piran images for one hotel', fontsize=12)

    for index, point in enumerate(components):
        point_size = 3

        if new_img_ids_vector[index] == '1':
            c = 'r'
        else:
            c = 'lightgray'
        ax.plot(point[0], point[1], marker='o', markersize=point_size, color=c)
    ax.grid()
    plt.show()

    # MDS

    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(features)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Kategorije', fontsize=12)
    ax.set_facecolor('#eaeaf2')

    for index, point in enumerate(X_transformed):
        point_size = 3
        ax.plot(point[0], point[1], marker='o', markersize=point_size, color=colors[categories_vector[index]])
    ax.grid()
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    ax.legend(custom_lines, colors.keys())
    plt.show()
    ax.grid(False)

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

    title = 'Kategorije'
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=20)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    for index, point in enumerate(components):
        color = colors[categories_vector[index]]
        ax.plot(point[0], point[1], marker='o', markersize=3, color=color)
    ax.grid()
    custom_lines = [Line2D([0], [0], color=value, lw=4) for value in colors.values()]
    names = ['Kranjska Gora', 'Piran']
    ax.legend(custom_lines, names)
    plt.show()




