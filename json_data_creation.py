from Data_Scraping.playbyplayscrape import GameScraper
from Data_Scraping.url_scraperv2 import URL_Scrape
import time
import sys
import json
import pandas as pd
import os
from play_classification import Play
from model_creation.model_constants import ModelConstants as MC
from datetime import datetime


sys.setrecursionlimit(10000)

def convert_time(time):
    """converting a time in the format '11:25.0' to the number of seconds left in the quatre"""

    time_obj = datetime.strptime(time[:-2], "%M:%S")
    return time_obj.second + (time_obj.minute * 60)

def write_season_json(base_url, num_games, team_code, season):

    """scrapes data for a season and writes it to a json file
    base_url --> the url for the season's data
    num_games --> the number of regular games played in the season (excluding playoffs)
    team_code --> e.g. MIL for Milwuakee (i.e. scraping all Milwuakee games for the specified season)
    season --> the season years in the format 2014_15 (for file names)
    """

    # BASE_URL = r"https://www.basketball-reference.com/teams/MIL/2022_games.html"
    # NUM_GAMES = 82

    data = {}

    url_scraper = URL_Scrape(base_url)

    game_count = 1
    for info in zip(url_scraper.get_game_urls(), url_scraper.get_game_metadata()):
        url = info[0]
        metadata = info[1]

        time.sleep(5)

        game_obj = GameScraper(url, metadata) # scraper and info holder object for a game
        game_obj.get_pbp()

        data[game_count] = {
            "metadata": metadata,
            "plays": game_obj.plays
        }

        print(f"data appended - game {game_count}/{num_games}")
        game_count += 1

    json_data = json.dumps(data, indent=4)
    filename = f"{team_code}_{season}_Season.json"

    try:  
        with open(filename, "w") as file:
            file.write(json_data)
    
        print(f"data successfully written to {filename}")

    except:
        raise Exception("Error when writing the json data")



def json_to_df_no_point_or_status_classification(filename, target_team):

    """
    Here in the play classification the point value and missed/made aren't included as these are put into separate features
    Thus it must always include good_or_bad and score_impact_no_penalty
    """

    with open(filename, "r") as file:
        json_data = json.load(file)

    col_ind = 0

    # note sub status only applies to the target_team
    # df = pd.DataFrame(columns=["game number", "time", "score_diff", "team", "play", "good_or_bad", "foul_count", "current_sub_status"])
    df = pd.DataFrame(columns=["game number", "time", "time_seconds", "score_diff", "good_or_bad", "foul_count", "score_impact", "time_since_last_score", "score_impact_no_penalty", "score_impact_no_opp_penalty", "team", "play", "current_sub_status"])


    for game_num in json_data:

        mistakes = 0 # aggregate count of mistakes
        fouls_count = 0 # aggregate count of fouls
        successes = 0 # aggregate count of successes
        time_since_last_score = None
        last_score_timestamp = None

        left_team = json_data[game_num]["metadata"]["left_team"]
        right_team = json_data[game_num]["metadata"]["right_team"]

        print(game_num)

        for play in json_data[game_num]["plays"]:

            play_obj = Play(
                text=play["play_description"],
                time=play["time"],
                score=play["score"],
                left_team=left_team,
                right_team=right_team,
                team=play["team"],
                target_team=target_team,
                game_num=game_num
            )

            play_obj.classify_play()

            time_seconds = convert_time(play["time"])

            if play_obj.type == "foul" and not play_obj.positive:
                fouls_count += 1

            if play_obj.type == "substitution" and play_obj.team == target_team:
                curr_label = 1
            else:
                curr_label = 0

            if play_obj.positive:
                good_or_bad = 1
                successes += 1
            else:
                good_or_bad = 0
                mistakes += 1

        
            score_impact = play_obj.play_point_impact
            score_impact_no_penalty = play_obj.play_point_impact_no_penalty
            score_impact_no_opp_penalty = play_obj.play_point_impact_no_opp_penalty

            if play_obj.is_shot and play_obj.positive and play_obj.team == play_obj.target_team: # made shot
                time_since_last_score = 0
                last_score_timestamp = time_seconds
            
            else:
                if last_score_timestamp == None:
                    time_since_last_score = None
                else:
                    time_since_last_score = last_score_timestamp - time_seconds

                        

            df.loc[col_ind] = [
                game_num,
                play["time"],
                time_seconds,
                play_obj.score_diff,
                good_or_bad,
                fouls_count,
                score_impact,
                time_since_last_score,
                score_impact_no_penalty,
                score_impact_no_opp_penalty,
                play["team"],
                play_obj.type,
                curr_label
            ]
            
            col_ind += 1

        
    df = pd.get_dummies(df, columns=["play"])

    df[f"future_sub_status"] = df["current_sub_status"].shift(-MC.FUTURE_PERIOD_PREDICT)

    return df

def json_to_df(filename, target_team):

    """
    json_to_df function for the new csv system where all features are included and ones not wanted are excluded in preprocessing.
    All the other json_to_df functions will be archived

    score_impact_no_penalty is like score_impact but there are no negative values (e.g. a miss would just give 0 instead of -2 or -3)
    score_impact_no_opp_penalty is like score_impact but if you miss you get negative but if the opponent scores its just 0
    """

    with open(filename, "r") as file:
        json_data = json.load(file)

    col_ind = 0

    # note sub status only applies to the target_team
    # df = pd.DataFrame(columns=["game number", "time", "score_diff", "team", "play", "good_or_bad", "foul_count", "current_sub_status"])
    df = pd.DataFrame(columns=["game number", "time", "time_seconds", "score_diff", "good_or_bad", "foul_count", "score_impact", "time_since_last_score", "score_impact_no_penalty", "score_impact_no_opp_penalty", "quatre_number", "play_value_literature", "point_value", "team", "play", "current_sub_status"])


    for game_num in json_data:
        
        quatre_number = 1
        mistakes = 0 # aggregate count of mistakes
        fouls_count = 0 # aggregate count of fouls
        successes = 0 # aggregate count of successes
        time_since_last_score = None
        last_score_timestamp = None
        prev_time = 720

        left_team = json_data[game_num]["metadata"]["left_team"]
        right_team = json_data[game_num]["metadata"]["right_team"]

        print(game_num)

        for play in json_data[game_num]["plays"]:

            play_obj = Play(
                text=play["play_description"],
                time=play["time"],
                score=play["score"],
                left_team=left_team,
                right_team=right_team,
                team=play["team"],
                target_team=target_team,
                game_num=game_num
            )

            play_obj.classify_play()

            time_seconds = convert_time(play["time"])

            if play_obj.type == "foul" and not play_obj.positive:
                fouls_count += 1

            if play_obj.type == "substitution" and play_obj.team == target_team:
                curr_label = 1
            else:
                curr_label = 0

            if play_obj.positive:
                good_or_bad = 1
                successes += 1
            else:
                good_or_bad = 0
                mistakes += 1

        
            score_impact = play_obj.play_point_impact
            score_impact_no_penalty = play_obj.play_point_impact_no_penalty
            score_impact_no_opp_penalty = play_obj.play_point_impact_no_opp_penalty

            if play_obj.is_shot and play_obj.positive and play_obj.team == play_obj.target_team: # made shot
                time_since_last_score = 0
                last_score_timestamp = time_seconds
            
            else:
                if last_score_timestamp == None:
                    time_since_last_score = None
                else:
                    time_since_last_score = last_score_timestamp - time_seconds

            if time_seconds > prev_time:
                quatre_number += 1


            df.loc[col_ind] = [
                game_num,
                play["time"],
                time_seconds,
                play_obj.score_diff,
                good_or_bad,
                fouls_count,
                score_impact,
                time_since_last_score,
                score_impact_no_penalty,
                score_impact_no_opp_penalty,
                quatre_number,
                play_obj.play_literature_pointvalue,
                play_obj.point_value,
                play["team"],
                play_obj.type,
                curr_label
            ]
            
            col_ind += 1
            prev_time = time_seconds

        
    df = pd.get_dummies(df, columns=["play"])

    df[f"future_sub_status"] = df["current_sub_status"].shift(-MC.FUTURE_PERIOD_PREDICT)

    return df

def save_df(df, csv_name):
    df.to_csv(f"Data\{csv_name}")

