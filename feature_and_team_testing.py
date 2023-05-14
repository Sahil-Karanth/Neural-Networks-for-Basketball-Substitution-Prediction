from statistics import median, mean
from webbrowser import get
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_multiple_files
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import seaborn as sns
import numpy as np
import time
from model_creation.model_constants import ModelConstants as MC
from model_creation.getting_training_data import get_file_paths, get_X_y_1_team, get_X_y_2_teams, get_X_y_3_teams
from keras_tuner import Hyperband
from scikeras.wrappers import KerasClassifier # allows GridSearchCV from sklearn to be used with keras
import pandas as pd
import matplotlib.pyplot as plt
import json
from itertools import combinations
from model_creation.getting_training_data import TEAMS


def make_a_model(X_train):

    """base function"""


    model = Sequential()

    model.add(LSTM(64, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(64, input_shape=(X_train.shape[1:])))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6)

    print("compiling model...")

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )

    print(model.summary())

    return model



def make_a_model_tuned(X_train):

    """makes a model with the results of the hyperparameter tuning"""

    model = Sequential()

    model.add(LSTM(64, input_shape=(X_train.shape[1:]), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(64, input_shape=(X_train.shape[1:])))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=1e-6)

    print("compiling model...")

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"]
    )

    print(model.summary())

    return model





def save_history_and_graph_model_teams_test(history, model_name):
    # summarize history for accuracy
    accuracy_figure = plt.figure()
    plt.plot(history.history['accuracy'])   
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"images/{MC.EPOCHS} Epochs Accuracy plot for {model_name}.png")
    # summarize history for loss
    loss_figure = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"images/{MC.EPOCHS} Epochs Loss plot for {model_name}.png")


    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)

    # save to json:  
    hist_json_file = f'{model_name}_history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f, indent=4)

def get_callbacks(early_stopping=False, early_stopping_monitor="val_accuracy", tensor_board=True):

    callbacks_lst = []
    # filepath = f"{MC.NAME} " + "{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
    if tensor_board:
        tensorboard = TensorBoard(log_dir=f"logs/{MC.NAME}")
        callbacks_lst.append(tensorboard)

    if early_stopping:
        early_stop = EarlyStopping(
            monitor=early_stopping_monitor,
            patience=20, # number of epochs to stop after if the monitor doesn't improve
            verbose=1,
            mode="max"
        )
        callbacks_lst.append(early_stop)

    return callbacks_lst


def teams_test(features: list):
    """testing networks for 1-3 teams and getting graphs"""

    for num_teams in range(1, 4):

        # BASE_PATH = os.getcwd()
        # num_teams = int(input("How many teams?: "))

        if num_teams == 1:
            X, y = get_X_y_1_team(added_features=features)
            teams = ["MIL"]

        elif num_teams == 2:
            X, y = get_X_y_2_teams(added_features=features)
            teams = ["MIL", "BOS"]

        elif num_teams == 3:
            X, y = get_X_y_3_teams(added_features=features)
            teams = ["MIL", "BOS", "MIA"]

        else:
            raise Exception("Bad user input")



        NAME = f"{MC.EPOCHS}_EPOCHS_{teams}_{MC.NAME}"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)



        print("X, y obtained")

        callbacks_lst = get_callbacks()

        # filepath = f"{MC.NAME} " + "{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
        # checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

        model = make_a_model()

        history = model.fit(
            X_train, y_train,
            batch_size=MC.BATCH_SIZE,
            epochs=MC.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks_lst
        )

        save_history_and_graph_model_teams_test(history, NAME)

        model.save_weights(f"saved_weights/{NAME}")

        time.sleep(5)


def save_histories_and_graph_models_feature_tests(histories, model_names, legend, folder="Individual Feature Tests"):

    """
    Takes a list of model histories and outputs 4 line graphs for each plot type each with a line for each model

    The plot types are 'Validation Accuracy', 'Accuracy', 'Validation Loss' or 'Loss'
    model_names and legend must be in the same order

    If graphing is False, only the json files will be saved (graphs won't be produced)
    """

    plot_types = ["Validation Accuracy", "Accuracy", "Validation Loss", "Loss"]

    for plot_type in plot_types:

        plt.title(f'Model {plot_type}')
        plt.ylabel(f'Feature {plot_type}')
        plt.xlabel('Epoch')

        if plot_type == "Validation Accuracy":
            target = "val_accuracy"
            loc = "upper left"
        
        elif plot_type == "Accuracy":
            target = plot_type.lower()
            loc = "upper left"

        elif plot_type == "Validation Loss":
            target = "val_loss"
            loc = "upper right"

        elif plot_type == "Loss":
            target = plot_type.lower()
            loc = "upper right"

        else:
            raise ValueError("Incorrect plot_type")
        


        count = 0
        for history, name in zip(histories, model_names):
            # summarize history for accuracy
            # plt.plot(history.history['val_accuracy'])
            plt.plot(history.history[target])   


            # convert the history.history dict to a pandas DataFrame:     
            hist_df = pd.DataFrame(history.history)

            # save to json:  
            hist_json_file = f'{folder}/{name}_history.json'
            with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f, indent=4)

            count += 1


        # plt.legend(legend, loc=loc)
        plt.legend(legend, bbox_to_anchor=(1.04, 1), loc="upper right")
        # plt.savefig(f"images/test_fig_1.png")
        plt.savefig(f"Feature Comparisons/4th time {MC.EPOCHS} Epochs {plot_type} plot for individual feature tests.png")

def save_histories(histories, model_names, folder="Team Combination Tests"):

    for history, name in zip(histories, model_names):
        hist_df = pd.DataFrame(history.history)

        # save to json:
        hist_json_file = f'{folder}/{name}_history.json'
        print(f"saving histories to {hist_json_file}")
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f, indent=4)

def baseline_feature_tests(num_teams, added_features_lst, legend, early_stopping=False, graphing=True):
    """
    trains and produces a graph showing model performance with each added feature individually

    added_features_lst is a list of lists where each sublist is the features added in a model being graphed
        e.g.     added_features_lst = [
                                        [],
                                        ["time_seconds"],
                                        ["score_diff"],
                                        ["good_or_bad"],
                                        ["foul_count"],
                                    ]

    graphing is a boolean variable which determines whether graphs will be produced along with the json files and saved weights

    if not graphing, legend is just a list of the names of features used in each model

    """

    num_teams_funcs = {
        1: get_X_y_1_team,
        2: get_X_y_2_teams,
        3: get_X_y_3_teams
    }

    histories = []

    for i,legend_name in enumerate(legend):

        print(f"DOING BASELINE TEST FOR {legend_name}")

        NAME = f"{MC.SEQ_LEN}-SEQ-{MC.FUTURE_PERIOD_PREDICT}-PRED-{MC.EPOCHS}-EPOCHS-{''.join(legend_name)}-DATAPATH-{added_features_lst[i]}-FEATURES_{num_teams}_teams_{time.time()}"

        X, y = num_teams_funcs[num_teams](added_features=added_features_lst[i], team1code="MIL", team2code="BOS")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


        print("X, y obtained")
        

        callbacks_lst = get_callbacks(early_stopping=early_stopping, tensor_board=True)

        model = make_a_model(X_train)

        history = model.fit(
            X_train, y_train,
            batch_size=MC.BATCH_SIZE,
            epochs=MC.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks_lst
        )

        histories.append(history)

        # model.save_weights(f"3rd_time-WEIGHTS_SAVED-FEATURE_TESTS-{MC.EPOCHS}-EPOCHS-{''.join(legend_name)}-{num_teams}-TEAMS-{time.time()}")

        time.sleep(5)

    model_names = [f"COMBINATION-TEST-{MC.EPOCHS}-EPOCHS-{''.join(j)}-{num_teams}-TEAMS-{time.time()}" for j in legend]

    if graphing:
        save_histories_and_graph_models_feature_tests(
            histories=histories,
            model_names=model_names,
            legend=legend,
        )

    else:
        save_histories(
            histories=histories,
            model_names=model_names
        )


def do_individual_feature_tests():

    num_teams = 2

    added_features_lst = [ # for each separate model
        ["good_or_bad", "point_value"],
        ["time_seconds", "good_or_bad", "point_value"],
        ["score_diff", "good_or_bad", "point_value"],
        ["foul_count", "good_or_bad", "point_value"],
        ["score_impact", "good_or_bad", "point_value"],
        ["time_since_last_score", "good_or_bad", "point_value"],
        ["score_impact_no_penalty", "good_or_bad", "point_value"],
        ["score_impact_no_opp_penalty", "good_or_bad", "point_value"],
        ["quatre_number", "good_or_bad", "point_value"],
        ["play_literature_pointvalue", "good_or_bad", "point_value"]
    ]

    legend = ["no added features", "time_seconds", "score_diff", "foul_count", "score_impact", "time_since_last_score", "score_impact_no_penalty", "score_impact_no_opp_penalty", "quatre_number", "play_literature_pointvalue"]

    baseline_feature_tests(
        num_teams=num_teams,
        legend=legend,
        added_features_lst=added_features_lst
    )


def feature_combination_tests(num_teams, legend, group_size):

    feature_combinations = [list(i) for i in list(combinations(legend, group_size))]

    for i in feature_combinations:
        i.extend(["good_or_bad", "point_value"])

    print(feature_combinations)
    baseline_feature_tests(
        num_teams=num_teams,
        added_features_lst=feature_combinations,
        legend=feature_combinations,
        graphing=False,
        early_stopping=True
    )


def do_combined_feature_tests(group_size):

    num_teams = 2

    legend = ["foul_count", "time_seconds", "quatre_number", "score_diff", "time_since_last_score"] # legend description for each sub list in added_features_lst

    feature_combination_tests(
        num_teams=num_teams,
        legend=legend,
        group_size=group_size
    )


# model_funcs = [
#     make_a_model,
#     make_a_model2,
#     make_a_model3,
#     make_a_model4,
#     make_a_model5,
#     make_a_model6,
# ]

# for func in model_funcs:

#     baseline_feature_tests(
#         num_teams=2,
#         added_features_lst=[["time_seconds", "good_or_bad", "point_value"]],
#         legend=["time_seconds"],
#         make_model_func=func
#     )



def graph_histories_manual():
    
    num_teams = 2
    plot_types = ["Validation Accuracy", "Training Accuracy", "Validation Loss", "Loss"]
    # legend = ["no added features", "play literature point value", "quatre number", "score difference", "score impact no opp penalty", "score impact no penalty", "score impact", "time_seconds", "time since last score"]
    legend = []

    for plot_type in plot_types:

        for json_history_path in os.listdir("Feature Comparisons/First Draft Data"):

            file = open(os.path.join(os.getcwd(), "Feature Comparisons", "First Draft Data", json_history_path))
            data = json.load(file)
            p1 = json_history_path.partition("EPOCHS-")
            legend_label = p1[-1].partition("-")[0]
            legend.append(legend_label)
            print(legend_label)

            file.close()

            plt.title(f'Individual Feature Ablation Study {plot_type}')
            plt.ylabel(f'{plot_type}')
            plt.xlabel('Epoch')

            plt.tight_layout()

            if plot_type == "Validation Accuracy":
                target = data["val_accuracy"]
                loc = "upper left"
            
            elif plot_type == "Training Accuracy":
                target = data["accuracy"]
                loc = "upper left"

            elif plot_type == "Validation Loss":
                target = data["val_loss"]
                loc = "upper right"

            elif plot_type == "Loss":
                target = data[plot_type.lower()]
                loc = "upper right"

            else:
                raise ValueError("Incorrect plot_type")
            
            target_keys = [int(i)+1 for i in target.keys()]

            plt.plot(target_keys, target.values())

        # plt.legend(legend, loc=loc)
        # plt.legend(legend, loc="lower right")

        plt.xlim(0, 100)
        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        plt.show()


# graph_histories_manual()

# baseline_feature_tests(
#     num_teams=2,
#     added_features_lst=[["time_seconds", "good_or_bad", "point_value"]],
#     legend=["time_seconds"],
#     make_model_func=make_a_model
# )


def team_combination_tests():

    histories = []
    names = []
    test_results = []

    for c in combinations(TEAMS.keys(), 2):
        print(f"teams used: {c}")

        added_features = ["good_or_bad", "point_value", "time_seconds", "score_diff", "time_since_last_score", "quatre_number"]
        X, y = get_X_y_2_teams(added_features, c[0], c[1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

        callbacks_lst = get_callbacks(early_stopping=False, tensor_board=True)

        model = make_a_model_tuned(X_train)

        history = model.fit(
            X_train, y_train,
            batch_size=MC.BATCH_SIZE,
            epochs=MC.EPOCHS,
            callbacks=callbacks_lst
        )

        histories.append(history)
        names.append(f"{'-'.join(c)}-TEAM_COMBINATION_TESTS_FINAL_OFFICIAL-{MC.EPOCHS}-EPOCHS")

        y_pred = model.predict(X_test)
        print(f"TEST RESULTS (loss, acc): {model.evaluate(x=X_test, y=y_test)}")

        test_results.append(model.evaluate(x=X_test, y=y_test))


    folder = "Team Combination Tests"
    save_histories(histories, names, folder)

    return test_results


    """
import numpy as np
import matplotlib.pyplot as plt

# Your one-dimensional data array
data = np.random.rand(50)

# Create a new figure and axis
fig, ax = plt.subplots()

# Create a horizontal boxplot
ax.boxplot(data, vert=False)

# Adjust the y-axis limits
# You can tweak the values (0.5 and 1.5) to control the whitespace around the boxplot
ax.set_ylim(0.5, 1.5)

# Display the plot
plt.show()
    """


def rank_comb_tests(filepath):

    accuracies = [] # (epoch num, val_acc, filename) tuple list for each file

    for filename in os.listdir(filepath):
        file = open(os.path.join(os.getcwd(), filepath, filename))
        data = json.load(file)
        
        val_acc_epoch = list(data["val_accuracy"].keys())[-1]
        # val_acc = list(data["val_accuracy"].values())[-1]

        num_average = 5 # number of last validation accuracies to average together

        tot = 0
        for i in range(num_average):
            tot += list(data["val_accuracy"].values())[-i]

        val_acc = tot / num_average

        accuracies.append((filename, val_acc_epoch, val_acc))

    accuracies.sort(key=lambda x: x[-1])

    # graph_accuracies(accuracies)

    return accuracies


def train_final_model(early_stopping=False):

    with open("training_data.npy", "rb") as file:
        X_train = np.load(file)
        y_train = np.load(file)

    with open("testing_data.npy", "rb") as file:
        X_test = np.load(file)
        y_test = np.load(file)

    callbacks_lst = get_callbacks(early_stopping=early_stopping, early_stopping_monitor="loss", tensor_board=True)

    model = make_a_model_tuned(X_train)

    history = model.fit(
        X_train, y_train,
        batch_size=MC.BATCH_SIZE,
        epochs=MC.EPOCHS,
        # validation_data=(X_val, y_val),
        callbacks=callbacks_lst
    )

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)

    name = f"FINAL_MODEL"

    hist_json_file = f'Final Model/{name}_history-{time.time()}.json'
    print(f"saving histories to {hist_json_file}")

    # saving model history
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f, indent=4)

    model.save_weights(f"Final Model/Final_Model_Weights-{time.time()}")

    y_pred = model.predict(X_test)
    print(f"TEST RESULTS (loss, acc): {model.evaluate(x=X_test, y=y_test)}")

    y_pred = np.round(y_pred).tolist()

    cm = confusion_matrix(y_test, y_pred)
    print("cm: ")
    print(cm)

    cm_normal = confusion_matrix(y_test, y_pred, normalize="pred")
    print("normalised cm: ")
    print(cm_normal)

    # save both confusion matrices
    np.save(f"Final Model/{name}_confusion_matrix-{time.time()}", cm)
    np.save(f"Final Model/{name}_confusion_matrix_normalised-{time.time()}", cm_normal)

    sns.heatmap(
        cm_normal,
        cmap="Reds",
        annot=True,
        cbar_kws={"orientation": "vertical"},
        xticklabels=["no sub", "sub"],
        yticklabels=["no sub", "sub"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()



def boxplot_team_comb_tests_on_validation_data(dir):

    plot_data = []

    for filename in os.listdir(dir):
        file = open(os.path.join(os.getcwd(), "Team Combination Tests", filename))
        data = json.load(file)
        val_acc = list(data["val_accuracy"].values())[-1]
        plot_data.append(val_acc)

    q3, q1 = np.percentile(plot_data, [75 ,25])
    iqr = q3 - q1
    median = np.median(plot_data)

    print(f"Interquartile range: {iqr}")
    print(f"Median range: {median}")

    plt.boxplot(plot_data, vert=False)

    # decrease whitespace above and below the boxplot
    plt.ylim(0.5, 1.5)

    plt.xlabel("Final Validation Accuracy (%)")
    plt.yticks([])
    plt.title("Team Combination Validation Accuracy Distribution")

    plt.show()


def boxplot_team_comb_tests_on_test_data(y_pred_lst):

    q3, q1 = np.percentile(y_pred_lst, [75 ,25])
    iqr = q3 - q1
    median = np.median(y_pred_lst)
    range = max(y_pred_lst) - min(y_pred_lst)

    fig = plt.figure(figsize=(5, 17), dpi=100)
    ax = plt.subplot(3,1,2)

    print(f"Interquartile range: {iqr}")
    print(f"Median: {median}")
    print(f"Range: {range}")

    ax.boxplot(y_pred_lst, vert=False, whis=0.8, positions=[0], widths=[0.4])

    plt.xlabel("Test Set Accuracy")
    plt.yticks([])

    plt.tight_layout() 

    plt.title("Team Combination Validation Accuracy Distribution")

    plt.show()


def get_hyperparameter_tuning_results():

    master_df = pd.DataFrame()
        
    for path in os.listdir("Hyperparameter Tuning Results"):

        df = pd.read_csv(f"Hyperparameter Tuning Results/{path}")
        df.drop(columns=["Unnamed: 0"], inplace=True)

        master_df = pd.concat([master_df, df])


    master_df.sort_values(by=["mean_test_score"], inplace=True)

    best = master_df.tail(3)

    print(best.values)


def get_substitution_percentage():

    tot = 0
    sub_tot = 0
    for path in os.listdir("Data"):
        if path[-3:] == "csv" and (path[:3] == "MIL" or path[:3] == "BOS"):
            print(path)
            df = pd.read_csv(f"Data/{path}")
            df.drop(columns=["Unnamed: 0"], inplace=True)

            num_subs = len(df[df["play_substitution"] == 1])
            
            tot += len(df)
            sub_tot += num_subs

    print(f"substitution play percentage: {sub_tot / tot}")


vals = [[0.6633591055870056, 0.5889003872871399], [0.6621266007423401, 0.5975049734115601], [0.6562353372573853, 0.5979066491127014], [0.6550483703613281, 0.6038461327552795], [0.6585378646850586, 0.6006057262420654], [0.653886616230011, 0.6044710278511047], [0.6570576429367065, 0.6115363836288452], [0.658962070941925, 0.6000180244445801], [0.6503621935844421, 0.6177112460136414], [0.6522728204727173, 0.6005586385726929], [0.6555275917053223, 0.6041309833526611], [0.6429237127304077, 0.6196797490119934], [0.6529147028923035, 0.6090772151947021], [0.6479453444480896, 0.6214084029197693], [0.6485758423805237, 0.6182273030281067]]

accs = [i[-1] for i in vals]

# boxplot_team_comb_tests_on_test_data(accs)

train_final_model()