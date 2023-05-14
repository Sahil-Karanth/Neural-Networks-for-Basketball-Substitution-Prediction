import pandas as pd
from collections import deque
import numpy as np
from datetime import timedelta
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from model_creation.model_constants import ModelConstants as MC


def str_time_to_obj(str_time):
    prev_time_lst = [float(x) for x in str_time.split(":")]
    prev_time = timedelta(seconds=prev_time_lst[1], minutes=prev_time_lst[0])

    return prev_time

def is_valid_sequence(sequence):

    """checks if a sequence goes over games or quatres"""
    
    valid_seq = True # flag for whether the sequence goes over games or quatres

    prev_game_num = sequence[0][1]
    # prev_time_lst = [float(x) for x in sequence[0][2].split(":")]
    # prev_time = timedelta(seconds=prev_time_lst[1], minutes=prev_time_lst[0])

    prev_time = str_time_to_obj(sequence[0][2])
    
    for vector in list(sequence)[1:]:
        curr_game_num = vector[1]

        # curr_time_lst = [float(x) for x in vector[2].split(":")]
        # curr_time = timedelta(seconds=curr_time_lst[1], minutes=curr_time_lst[0])

        curr_time = str_time_to_obj(vector[2])

        if (curr_time > prev_time) or (curr_game_num != prev_game_num): # invalid sequence
            valid_seq = False
            break

        prev_game_num = curr_game_num
        prev_time = curr_time


    return valid_seq

def convert_time(time):
    """converting a time in the format '11:25.0' to the number of seconds left in the quatre"""

    time_obj = datetime.strptime(time[:-2], "%M:%S")
    return time_obj.second + (time_obj.minute * 60)

def remove_invalid_label_sequences(sequential_data: list):
    
    """removes sequences whose label goes over games or quatres. Assumes no sequences go over games or quatres"""

    filtered_sequences = []
    for i in range(len(sequential_data)-10):

        label = sequential_data[i][1]
        seq = sequential_data[i][0] # excluding the label as we don't need it
        
        curr_time = str_time_to_obj(seq[-1][2])
        curr_game_num = seq[0][1] # the 0 indexes a play in the sequence and could be any of them I just chose the first
        
        future_seq = sequential_data[i+10][0] # the "future" sequence contianing the label as a curr_sub_status
        future_game_num = future_seq[0][1]
        future_time = str_time_to_obj(future_seq[-1][2])

        if (future_game_num == curr_game_num) or (future_time <= curr_time): # valid sequence
            filtered_sequences.append([seq, label])

    return filtered_sequences

def preprocess_multiple_files(data_paths: list, target_team, added_features: list):
    """
    data_paths is a list of csv files
    target_team is the team you're looking at in each game
    added_features is a list of features being added e.g. ["score_diff", "time_seconds", "foul_count", "good_or_bad"]
    
    feature_indexes = {
    "time_seconds": 3,
    "score_diff": 4,
    "good_or_bad": 5,
    "foul_count": 6,
    "score_impact": 7,
    "time_since_last_score": 8,
    "score_impact_no_penalty": 9,
    "score_impact_no_opp_penalty": 10,
    "quatre_number": 11,
    "play_literature_pointvalue": 12,
    }

    """

    # the keys (integers) are the indexes in the csv (and thus in df.columns) of the optional features
    feature_indexes = {
        "time_seconds": 3,
        "score_diff": 4,
        "good_or_bad": 5,
        "foul_count": 6,
        "score_impact": 7,
        "time_since_last_score": 8,
        "score_impact_no_penalty": 9,
        "score_impact_no_opp_penalty": 10,
        "quatre_number": 11,
        "play_literature_pointvalue": 12,
        "point_value": 13
    }

    # list of csv indexes to not include as features
    # 'Unnamed: 0', 'game number' and 'time' (0,1,2) will always be dropped
    indexes_to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # game number and time are needed to determine if a sequence is valid however so dropping the columns in this list shouldn't be done till the very end
    
    print(f"FEATURES ADDED: {added_features}")

    for feature in added_features:
        if feature not in feature_indexes.keys():
            raise ValueError("A feature in added_features is not valid. Spelling Issue?")

        drop_ind = feature_indexes[feature]
        indexes_to_drop.remove(drop_ind)
        print(f"successfully popped {feature}")
        print(indexes_to_drop)


    SEQ_LEN = MC.SEQ_LEN

    scale_cols = []
    for feature in added_features:
        if feature in ["score_diff", "time_seconds", "foul_count", "score_impact", "quatre_number", "point_value", "score_impact", "time_since_last_score", "score_impact_no_penalty", "score_impact_no_opp_penalty", "play_literature_pointvalue"]:
            scale_cols.append(feature)

    print(f"COLS TO SCALE: {scale_cols}")

    X = []
    y = []

    for data_path in data_paths:

        print(f"getting data from {data_path}")

        df = pd.read_csv(data_path)

        df = df.drop(['play_substitution','current_sub_status', "play_misses 3-pt layup", "play_makes 3-pt layup"], axis=1, errors='ignore')

        # MOVED TO THE CSV FILE SO THE COL IS ALREADY THERE
        # df.insert(3, "time_seconds", df["time"].apply(convert_time)) # converts time format to seconds

        # scaling


        scaler = StandardScaler()
        for col in df.columns:
            if col in scale_cols:
                arr = df[col].values.reshape(-1,1)
                df[col] = scaler.fit_transform(arr)
                print(f"SCALING {col}")
                print(df[col]) # should now be scaled


        df.loc[df["team"] == target_team, "team"] = 1
        df.loc[df["team"] != 1, "team"] = 0

        df.dropna(inplace=True)
        
        print("MAKING SEQUENTIAL DATA")
        sequential_data = []
        prev_plays = deque(maxlen=SEQ_LEN)
        for i in df.values:
            vector = [n for n in i[:-1]]
            prev_plays.append(vector)

            if len(prev_plays) == SEQ_LEN and is_valid_sequence(prev_plays):
                sequential_data.append([np.array(prev_plays), i[-1]])

        sequential_data = remove_invalid_label_sequences(sequential_data)

        random.shuffle(sequential_data)

        subs = []
        non_subs = []

        for seq, label in sequential_data:
            if int(label) == 1:
                subs.append([seq, label])
            elif int(label) == 0:
                non_subs.append([seq, label])

        random.shuffle(subs)
        random.shuffle(non_subs)

        lower = min(len(subs), len(non_subs))

        subs = subs[:lower]
        non_subs = non_subs[:lower]

        sequential_data = subs + non_subs

        random.shuffle(sequential_data)

        # excluding the features in indexes_to_drop
        print(f"EXCLUDING INDEXES {indexes_to_drop}")
        first_iter = True
        for seq, label in sequential_data:


            new_seq = [np.delete(i, indexes_to_drop) for i in seq]
            
            if first_iter:
                print("First sequence for reference:")
                print(new_seq[0])
                first_iter = False

            X.append(new_seq)
            y.append(label)


    print(f"returning X and y from preprocess_multiple_files for {target_team}")

    print(f"1 LABEL COUNT: {y.count(1)}")
    print(f"0 LABEL COUNT: {y.count(0)}")


    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")

    # print([i for i in list(y)])

    return (X, y)

