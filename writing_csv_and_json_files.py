from json_data_creation import json_to_df, save_df, write_season_json
import os

BASE_PATH = os.getcwd()

def make_binary_json(team_code):

    """writes the json for the specified seasons with binary features (1st iteration)"""

    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

    denver_num_games = {
        2015: 82,
        2016: 82,
        2017: 82,
        2018: 82,
        2019: 73,
        2020: 72,
        2021: 82
    }

    utah_num_games = {
        2015: 82,
        2016: 82,
        2017: 82,
        2018: 82,
        2019: 72,
        2020: 72,
        2021: 82
    }

    
    if team_code == "DEN":
        dict_ = denver_num_games

    elif team_code == "UTA":
        dict_ = utah_num_games


    for year in years:

        BASE_URL = f"https://www.basketball-reference.com/teams/{team_code}/{year}_games.html"

        print(f"WRITING {team_code} DATA FOR {year}")

        write_season_json(
            base_url=BASE_URL,
            num_games=dict_[year],
            team_code=team_code,
            season=f"{year}_{year+1}"
        )


def write_csvs_from_json(team_code_lst, team_name_lst):

    years_lst = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

    for team_code, team_name in zip(team_code_lst, team_name_lst):

        json_names = [os.path.join(BASE_PATH, "Data", f"{team_code}_{year}_{year+1}_Season.json") for year in years_lst]

        for i, j_file in enumerate(json_names):

            csv_name = f"{team_code}_{years_lst[i]}_{years_lst[i]+1}.csv"
            if os.path.isfile(f"Data/{csv_name}"):
                print(f"{csv_name} already exists. Skipping...")
                continue
            print(f"writing {j_file} to {csv_name}")
            df = json_to_df(j_file, target_team=team_name)
            save_df(df, csv_name)





write_csvs_from_json(
    ["DAL", "UTA", "DEN"],
    ["Dallas", "Utah", "Denver"],
)
