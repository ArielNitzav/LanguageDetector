import Preprocess
import Analysis

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pickle


def predict_lang(s):
    with open('neigh.pkl', 'rb') as file:
        neigh = pickle.load(file)
    with open('char_set.pkl', 'rb') as file:
        char_set = pickle.load(file)

    s = Preprocess.string_stripper(s)
    hist = Analysis.char_hist(s, char_set)
    hist_array = np.array([hist])

    return neigh.predict(hist_array)[0]