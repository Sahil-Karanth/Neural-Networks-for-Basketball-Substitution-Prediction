import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras_tuner import Hyperband, HyperModel
from scikeras.wrappers import KerasClassifier # allows GridSearchCV from sklearn to be used with keras


class LSTM_HyperModel(HyperModel):

    def __init__(self, min_layers, max_layers, X_shape):
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.X_shape = X_shape

    def build(self, hp):
        """
        hyperparameters tuned in this function:
        1) number of LSTM layers
        2) number of Dense layers
        3) number of neurons in layers
        4) learning rate
        """

        model = Sequential()

        # LSTM layers
        for i in range(hp.Int("num_layers", self.min_layers, self.max_layers)):
            
            hp_units = hp.Int("units_" + str(i), min_value=16, max_value=512, step=32)

            model.add(LSTM(units=hp_units, input_shape=(self.X_shape), return_sequences=True, recurrent_regularizer="l2"))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
        
        # Dense layer(s)
        for i in range(hp.Int("num_layers", 0, 2)):
            hp_units = hp.Int("units_" + str(i), min_value=16, max_value=32, step=16)

            model.add(Dense(units=hp_units, activation="relu"))
            model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(1, activation="sigmoid"))

        # Optimiser
        hp_learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])
        opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate, weight_decay=1e-6)

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=opt,
            metrics=["accuracy"]
        )

        return model






def build_model_keras(hp, min_layers, max_layers, X_shape):

    """
    hyperparameters tuned in this function:
    1) number of LSTM layers
    2) number of Dense layers
    3) number of neurons in layers
    4) learning rate
    """

    model = Sequential()

    # LSTM layers
    for i in range(hp.Int("num_layers", min_layers, max_layers)):
        
        hp_units = hp.Int("units_" + str(i), min_value=16, max_value=512, step=32)

        model.add(LSTM(units=hp_units, input_shape=(X_shape), return_sequences=True, recurrent_regularizer="l2"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    
    # Dense layer(s)
    for i in range(hp.Int("num_layers", 0, 2)):
        hp_units = hp.Int("units_" + str(i), min_value=16, max_value=32, step=16)

        model.add(Dense(units=hp_units, activation="relu"))
        model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Optimiser
    hp_learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])
    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate, weight_decay=1e-6)

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )

    return model

