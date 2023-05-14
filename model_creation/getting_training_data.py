from sklearn.model_selection import train_test_split
from preprocessing import preprocess_multiple_files
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
from model_creation.model_constants import ModelConstants as MC
from keras_tuner import Hyperband, HyperModel
from scikeras.wrappers import KerasClassifier # allows GridSearchCV from sklearn to be used with keras


BASE_PATH = os.getcwd()

TEAMS = {
"MIL": "Milwaukee",
"BOS": "Boston",
"MIA": "Miami",
"DAL": "Dallas",
"UTA": "Utah",
"Den": "Denver"
}

def get_file_paths_old(team_code, file_name_group):

    """file name group is the name of the iteration of file being used e.g. PLAY_CLASSIFICATION, PLAY_CLASSIFICATION_SCOREDIFF etc"""

    file_paths = [os.path.join(BASE_PATH, "Data", f"{team_code}_{year}_{year+1}_{file_name_group}.csv") for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021]]
    return file_paths

def get_file_paths(team_code):
    file_paths = [os.path.join(BASE_PATH, "Data", f"{team_code}_{year}_{year+1}.csv") for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021]]
    return file_paths


def get_X_y_2_teams(added_features, team1code, team2code):

    MIL_file_paths = get_file_paths(team1code)
    BOS_file_paths = get_file_paths(team2code)

    X_1, y_1 = preprocess_multiple_files(MIL_file_paths, target_team=TEAMS[team1code], added_features=added_features)
    X_2, y_2 = preprocess_multiple_files(BOS_file_paths, target_team=TEAMS[team2code], added_features=added_features)

    X = np.concatenate((X_1, X_2))
    y = np.concatenate((y_1, y_2))

    return (X, y)

def get_X_y_3_teams(added_features):

    MIL_file_paths = get_file_paths("MIL")
    BOS_file_paths = get_file_paths("BOS")
    MIA_file_paths = get_file_paths("MIA")

    X_MIL, y_MIL = preprocess_multiple_files(MIL_file_paths, target_team="Milwaukee", added_features=added_features)
    X_BOS, y_BOS = preprocess_multiple_files(BOS_file_paths, target_team="Boston", added_features=added_features)
    X_MIA, y_MIA = preprocess_multiple_files(MIA_file_paths, target_team="Miami", added_features=added_features)


    X = np.concatenate((X_MIL, X_BOS, X_MIA))
    y = np.concatenate((y_MIL, y_BOS, y_MIA))

    return (X, y)

def get_X_y_1_team(added_features):

    MIL_file_paths = get_file_paths("MIL")
    # BOS_file_paths = get_file_paths("BOS", file_name_group)
    # MIA_file_paths = get_file_paths("MIA")

    X_MIL, y_MIL = preprocess_multiple_files(MIL_file_paths, target_team="Milwaukee", added_features=added_features)
    # X_BOS, y_BOS = preprocess_multiple_files(BOS_file_paths, target_team="Boston", added_features=added_features)
    # X_MIA, y_MIA = preprocess_multiple_files(MIA_file_paths, target_team="Miami")


    X = X_MIL
    y = y_MIL

    return (X, y)
