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





def do_GridSearch(X, y):
    
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


    for j in range(10):
        search.fit(X_train, y_train)
        save_progress(search, f'checkpoint_{j}.pkl', f"RandomSearch_crossval_results_iteration_{j}.csv")

