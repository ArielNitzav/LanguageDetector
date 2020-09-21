import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle


def list_to_pd(sentence_list, df):
    append_df = pd.DataFrame(sentence_list, columns=["Sentence", "Language"])
    df = df.append(append_df)

    return df


def char_hist(sentence, char_set):
    hist_dict = {}
    for char in char_set:
        hist_dict[char] = sentence.count(char)
    hist = np.array(list(hist_dict.values()))
    hist = hist / hist.sum()

    return hist


def df_hist_col(df, char_set):
    df["Histogram"] = df.apply(lambda x: char_hist(x.Sentence, char_set), axis=1)
    return df


def pca_plotter(data_df, colors):
    matrix = np.vstack(data_df.Histogram)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(matrix)
    principal_df = pd.DataFrame(principal_components, columns=['pc1', 'pc2'])

    pca_data_df = pd.concat([data_df[['Language']], principal_df], axis=1, ignore_index=True)
    pca_data_df.columns = ['Language', 'pc1', 'pc2']

    fig, ax = plt.subplots()
    ax.scatter(pca_data_df['pc1'], pca_data_df['pc2'], c=pca_data_df['Language'].apply(lambda x: colors[x]))

    return ax

