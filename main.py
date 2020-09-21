import Preprocess
import Analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pickle

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

lang_list = ['Bulgarian', 'Czech', 'Danish', 'Estonian', 'Finnish', 'French', 'German', 'Greek', 'Hungarian', 'Italian',
             'Spanish', 'Swedish', 'English', 'Latvian', 'Lithuanian', 'Polish', 'Portuguese', 'Romanian', 'Slovene', 'Dutch']

char_set = []
data_df = pd.DataFrame(columns=["Sentence", "Language"])


for language in lang_list:
    sentence_list, char_set = Preprocess.sent_preprocess("Data/{}.txt".format(language), 200, 10000, language, char_set)
    data_df = Analysis.list_to_pd(sentence_list, data_df)

data_df = Analysis.df_hist_col(data_df, char_set).reset_index(drop=True)
colors = {'Bulgarian': 'green', 'Czech': 'blue', 'Danish': 'pink', 'Estonian': 'aquamarine', 'Finnish': 'purple', 'French': 'deeppink',
          'German': 'gold', 'Greek': 'deepskyblue', 'Hungarian': 'orange', 'Italian':'darkgreen', 'Spanish': 'yellow',
          'Swedish': 'slategrey', 'English': 'black', 'Latvian': 'fuchsia', 'Lithuanian': 'pink', 'Polish': 'lightcoral',
          'Portuguese': 'orangered', 'Romanian': 'lemonchiffon', 'Slovene': 'azure', 'Dutch': 'peru'}

# ax = Analysis.pca_plotter(data_df, colors)

train, test = train_test_split(data_df, test_size=0.2)
x_train = np.vstack(train.Histogram)
y_train = np.array(train.Language)
x_test = np.vstack(test.Histogram)
y_test = np.array(test.Language)

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(x_train, y_train)
predictions = neigh.predict(x_test)
test = pd.concat([test, pd.DataFrame(predictions, columns=["Predictions"])])

accuracy = neigh.score(x_test, y_test)
print(accuracy)

'''
pkl_filename = "neigh.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(neigh, file)

pkl_filename = "char_set.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(char_set, file)
'''

kmeans = KMeans(n_clusters=len(lang_list))
kmeans.fit(x_train)
predictions = kmeans.predict(x_test)

pkl_filename = "predictions.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(predictions, file)

pkl_filename = "labels.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(y_test, file)