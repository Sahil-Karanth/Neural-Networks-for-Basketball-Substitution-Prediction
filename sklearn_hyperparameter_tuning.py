from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from preprocessing import preprocess_multiple_files
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
import pickle
import os
from skopt import load
import numpy as np
import time
from model_creation.model_constants import ModelConstants as MC
from skopt.callbacks import DeltaYStopper
from model_creation.getting_training_data import get_file_paths, get_X_y_1_team, get_X_y_2_teams, get_X_y_3_teams
from keras_tuner import Hyperband
from scikeras.wrappers import KerasClassifier # allows GridSearchCV from sklearn to be used with keras
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import CheckpointSaver

def build_model_sk(X_shape, num_lstm_layers, num_dense_layers, lstm_units, dense_units, learning_rate):
    """
    building the model for hyperparameter tuning with sklearn's GridSearchCV
    
    hyperparameters tuned in this function:
    1) number of LSTM layers
    2) number of Dense layers
    3) number of neurons in layers
    4) learning rate
    """

    model = Sequential()

    # LSTM layers
    return_sequences = True
    for i in range(num_lstm_layers):

        if i == num_lstm_layers - 1:
            return_sequences = False
        
        model.add(LSTM(units=lstm_units, input_shape=(X_shape), return_sequences=return_sequences, recurrent_regularizer="l2"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    
    # Dense layer(s)
    for _ in range(num_dense_layers):

        model.add(Dense(units=dense_units, activation="relu"))
        model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Optimiser
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=1e-6)

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )

    return model


def do_fullscale_GridSearch(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = KerasClassifier(model=build_model_sk, epochs=500)

    param_grid = { # for the full scale grid search
    "num_lstm_layers": [1, 2, 3],
    "num_dense_layers": [0, 1, 2],
    "lstm_units": [16, 32, 64, 128],
    "dense_units": [16, 32],
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "batch_size": [16, 32],
    # "epochs": [300, 500, 800, 1000]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=3,
        scoring=["accuracy"],
        refit=False
        # return_train_score=True
    )

    result = grid.fit(X_train, y_train)

    df = pd.DataFrame(grid.csv_results_)
    df.to_csv("GridSearch_crossval_results.csv")


def do_small_scale_GridSearch(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    tensorboard = TensorBoard(log_dir=f"logs/{MC.NAME}")

    filepath = f"{MC.NAME} " + "{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones


    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=20, # number of epochs to stop after if the monitor doesn't improve
        verbose=1,
        mode="max"
    )


    num_lstm_layers = [2]
    num_dense_layers = [1,2]
    lstm_units = [64]
    dense_units = [32]
    learning_rate = [1e-4]
    # batch_size = [16, 24,32]

    model = KerasClassifier(
        model=build_model_sk,
        epochs=MC.EPOCHS,
        batch_size=MC.BATCH_SIZE,
        num_lstm_layers=num_lstm_layers,
        num_dense_layers=num_dense_layers,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        X_shape=X_train.shape[1:],
        # callbacks=[tensorboard]
    )


    param_grid = {
    "num_lstm_layers": num_lstm_layers,
    "num_dense_layers": num_dense_layers,
    "lstm_units": lstm_units,
    "dense_units": dense_units,
    "learning_rate": learning_rate,
    # "batch_size": batch_size,
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=2,
        verbose=3,
        scoring=["accuracy"],
        refit=False
        # return_train_score=True
    )

    grid.fit(X_train, y_train)

    df = pd.DataFrame(grid.cv_results_)
    df.to_csv("GridSearch_crossval_results.csv")


def save_progress(model, pkl_filename, csv_filename):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(model.cv_results_, f)

    df = pd.DataFrame(model.cv_results_)
    df.to_csv(csv_filename)

def do_RandomSearch():
    
    with open("training_data.npy", "rb") as file:
        X_train = np.load(file)
        y_train = np.load(file)

    tensorboard = TensorBoard(log_dir=f"logs/{MC.NAME}")

    num_lstm_layers = [2, 3]
    num_dense_layers = [1, 2, 3]
    lstm_units = [32, 64, 128]
    dense_units = [32, 64, 128]
    learning_rate = [1e-5, 1e-4]
    batch_size = [32]

    model = KerasClassifier(
        model=build_model_sk,
        epochs=MC.EPOCHS,
        batch_size=MC.BATCH_SIZE,
        num_lstm_layers=num_lstm_layers,
        num_dense_layers=num_dense_layers,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        X_shape=X_train.shape[1:],
        # callbacks=[tensorboard]
    )


    param_grid = {
    "num_lstm_layers": num_lstm_layers,
    "num_dense_layers": num_dense_layers,
    "lstm_units": lstm_units,
    "dense_units": dense_units,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=2,
        scoring="accuracy",
        refit=True,
        cv=5,
        verbose=3
    )


    for j in range(20):
        search.fit(X_train, y_train)
        save_progress(search, f'checkpoint_{j}.pkl', f"RandomSearch_crossval_results_iteration_{j}.csv")

    
    # print(f'Best Score: {search.best_score_}')
    # print(f'Best Hyperparameters: {search.best_params_}')

    # df = pd.DataFrame(search.cv_results_)
    # df.to_csv("RandomSearch_crossval_results.csv")

def time_limit_callback(search):
    time_elapsed = time.time() - start_time
    if time_elapsed > 60:  # 4 days in seconds
        raise StopIteration()  # Stop search
    return False


class TimeStopper:

    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = time.time()

    def __call__(self, res):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_seconds:
            return True
        return False


def do_BayesSearch():
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    with open("training_data.npy", "rb") as file:
        X_train = np.load(file)
        y_train = np.load(file)

    tensorboard = TensorBoard(log_dir=f"logs/{MC.NAME}")

    num_lstm_layers = Integer(2, 3)
    num_dense_layers = Integer(1, 2)
    lstm_units = Categorical([32, 64, 128])
    dense_units = Categorical([32, 64, 128])
    learning_rate = Categorical([1e-5, 1e-4])
    batch_size = Categorical([24, 32])

    model = KerasClassifier(
        model=build_model_sk,
        epochs=MC.EPOCHS,
        batch_size=MC.BATCH_SIZE,
        num_lstm_layers=num_lstm_layers,
        num_dense_layers=num_dense_layers,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        X_shape=X_train.shape[1:],
        # callbacks=[tensorboard]
    )


    param_grid = {
    "num_lstm_layers": num_lstm_layers,
    "num_dense_layers": num_dense_layers,
    "lstm_units": lstm_units,
    "dense_units": dense_units,
    "learning_rate": learning_rate,
    "batch_size": batch_size
    }

    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_grid,
        n_iter=144, # the number of configuration tried
        scoring="accuracy",
        refit=True,
        cv=5,
        verbose=3
    )

    
    start_time = time.time()
    time_budget = 3 * 24 * 60 * 60 # 4 days in seconds

    checkpoint_saver = CheckpointSaver("./hyperparameter_search_ACTUAL_two.pkl")

    search.fit(X_train, y_train, callback=[checkpoint_saver, TimeStopper(time_budget)])
    
    print(f'Best Score: {search.best_score_}')
    print(f'Best Hyperparameters: {search.best_params_}')

    df = pd.DataFrame(search.cv_results_)
    df.to_csv("RandomSearch_crossval_results.csv")



def continue_BayesSearch():

    with open("training_data.npy", "rb") as file:
        X_train = np.load(file)
        y_train = np.load(file)

    # Load the search
    with open('hyperparameter_search_ACTUAL_one.pkl', 'rb') as f:
        search = joblib.load(f)


    num_lstm_layers = Integer(2, 3)
    num_dense_layers = Integer(1, 2)
    lstm_units = Categorical([32, 64, 128])
    dense_units = Categorical([32, 64, 128])
    learning_rate = Categorical([1e-5, 1e-4])
    batch_size = Categorical([24, 32])

    model = KerasClassifier(
        model=build_model_sk,
        epochs=MC.EPOCHS,
        batch_size=MC.BATCH_SIZE,
        num_lstm_layers=num_lstm_layers,
        num_dense_layers=num_dense_layers,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        X_shape=X_train.shape[1:],
        # callbacks=[tensorboard]
    )

    param_grid = {
    "num_lstm_layers": num_lstm_layers,
    "num_dense_layers": num_dense_layers,
    "lstm_units": lstm_units,
    "dense_units": dense_units,
    "learning_rate": learning_rate,
    "batch_size": batch_size
    }

    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_grid,
        n_iter=1, # the number of configuration tried
        scoring="accuracy",
        refit=True,
        cv=5,
        verbose=3
    )

    
    start_time = time.time()
    # time_budget = 24 * 60 * 60 # 4 days in seconds
    time_budget = 60

    checkpoint_saver = CheckpointSaver("./hyperparameter_search_ACTUAL_two.pkl")

    search.fit(X_train, y_train, callback=[checkpoint_saver, TimeStopper(time_budget)])
    
    print(f'Best Score: {search.best_score_}')
    print(f'Best Hyperparameters: {search.best_params_}')

    df = pd.DataFrame(search.cv_results_)
    df.to_csv("RandomSearch_crossval_results.csv")



do_RandomSearch()

# X, y = get_X_y_2_teams(["point_value", "good_or_bad", "score_diff", "time_seconds", "time_since_last_score", "quatre_number"], team1code="MIL", team2code="BOS")


# with open("training_data.npy", "wb") as file:
#     np.save(file, X_train)
#     np.save(file, y_train)


# with open("testing_data.npy", "wb") as file:
#     np.save(file, X_test)
#     np.save(file, y_test)
