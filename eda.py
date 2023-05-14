import os
from statistics import mean, mode, stdev
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# NOTE for some reason I called the data 2014_15 when it's actually MIL the 2021_22 data

def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def step_hist(base_season_path, num_games, title):
    """outputs a histogram showing the distribution of the time between plays/steps
    requires a folder named 'base_season_path' containing all the game data csv files"""
    
    time_diff_lst = []

    for i in range(1, num_games+1):
        print(f"game {i}")
        csv_file = os.path.join(base_season_path, f"{base_season_path}_game_{i}")
        prev_time = None

        with open(csv_file, 'r', encoding="utf8") as file:
            next(file) # skips first line of the file (column headers)
            for line in file:
                play_time = [float(x) for x in line.split(",")[-5].split(":")]
                time_obj = timedelta(minutes=play_time[0], seconds=play_time[1])

                if prev_time and prev_time >= time_obj: # not going over quatres
                    time_diff = prev_time - time_obj
                    time_diff_lst.append(time_diff.seconds)

                prev_time = time_obj

    mean_val = mean(time_diff_lst)
    mode_val = mode(time_diff_lst)

    print(f"Mean single step time difference: {mean_val}")
    print(f"Mode single step time difference: {mode_val}")
    print(f"the mode accounts for {(time_diff_lst.count(mode_val) / len(time_diff_lst)) * 100}% of the data")

    plt.hist(time_diff_lst, bins=len(set(time_diff_lst)), color="green", histtype='bar', ec="black")
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("frequency")
    plt.show()



def n_step_hist(base_season_path, num_games, chunk_size, title):
    """outputs a histogram showing the distribution of the time between n plays/steps"""

    time_diff_lst = []

    for i in range(1, num_games+1):
        print(f"game {i}")
        csv_file = os.path.join(base_season_path, f"{base_season_path}_game_{i}")

        with open(csv_file, 'r', encoding="utf8") as file:
            next(file) # skips first line of the file (column headers)

            lines = [x.split(",")[-5] for x in file.readlines()]
            chunked = [[x[0],x[-1]] for x in divide_chunks(lines, chunk_size)]

            for j in chunked:
                t1 = timedelta(minutes=float(j[0].split(":")[0]), seconds=float(j[0].split(":")[1]))
                t2 = timedelta(minutes=float(j[1].split(":")[0]), seconds=float(j[1].split(":")[1]))

                if t1 >= t2: # doesn't go over quatres
                    time_diff = t1 - t2
                    time_diff_lst.append(time_diff.seconds)

    print(f"Mean 10 step time difference: {mean(time_diff_lst)}")
    print(f"standard deivation: {stdev(time_diff_lst)}")

    plt.hist(time_diff_lst, bins=30, color="green", histtype='bar', ec="black")
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("frequency")
    plt.show()




n_step_hist(
    base_season_path="2014_15",
    num_games=82,
    chunk_size=10,
    title="Time between 10 plays over the 2021/22 Milwaukee Bucks season"
)


step_hist(
    base_season_path="2014_15",
    num_games=82,
    title="Time between plays over the 2021/22 Milwaukee Bucks season"
)


# mean 10 step is 54.194244604316545
# standard deivation 10 step is 21.589710367883658

# mean single step is 6.1571443798966055
# mode single step is 0
    # the mode accounts for 30.82662687203539% of the data