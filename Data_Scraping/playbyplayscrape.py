import requests
from bs4 import BeautifulSoup


class Play: # data object for a single play
    def __init__(self, time, team, play_description, score):
        self.time = time
        self.team = team
        self.play = play_description
        self.score = score

    def __str__(self):
        return f"""
        Time: {self.time}
        Score: {self.score}
        Play team: {self.team}
        Play: {self.play}
        """


# scraper for one game
class GameScraper:

    NULL_CHAR = "\xa0"

    def __init__(self, base_url, metadata):
        self.r = requests.get(base_url)
        self.soup = BeautifulSoup(self.r.text, "lxml")
        self.raw_pbp = self.soup.find("table", id="pbp").findAll("tr") # iterable with each table column
        self.pbp = [i.text.split("\n") for i in self.raw_pbp]
        self.left_team = self.pbp[1][2]
        self.right_team = self.pbp[1][6]
        self.plays = []
        self.metadata = metadata
        self.metadata["left_team"] = self.left_team
        self.metadata["right_team"] = self.right_team

    def get_pbp(self):
        for j in range(len(self.raw_pbp)):
            row = [i.text for i in self.raw_pbp[j].findAll("td")] # row in the play-by-play table
            if len(row) == 6: # play row

                # team and play deciding code
                if row[1] == GameScraper.NULL_CHAR:
                    team = self.right_team
                    play_desc = row[-1]
                elif row[-1] == GameScraper.NULL_CHAR:
                    team = self.left_team
                    play_desc = row[1]
                else:
                    raise Exception("Error in categorising the team that a play belongs to")
            
                # play_obj = Play(
                #     team=team,
                #     time=row[0],
                #     score=row[3],
                #     play_description=play_desc
                # )

                play_dict = {
                    "team": team,
                    "time": row[0],
                    "score": row[3],
                    "play_description": play_desc
                }

                self.plays.append(play_dict)