import pandas as pd
from datetime import datetime

class Play:

    PLAY_TYPES = [
        "made jumpshot",
        "missed jumpshot",
        "made layup",
        "missed layup",
        "defensive rebound",
        "offensive rebound"
        "turnover",
        "foul",
        "made free throw",
        "missed free throw",
        "substitution"
    ]

    def __init__(self, text, time, score, team, left_team, right_team, target_team, game_num=None):
        self.text = text
        self.time = self.convert_time(time)

        self.left_team = left_team
        self.right_team = right_team
        self.target_team = target_team
        self.score_diff = self.calc_score_diff(score)
        
        self.team = team
        self.game_num = game_num

        self.split_text = text.split()
        self.positive = True
        self.play_point_impact = 0 # default is 0 as any non-shooting play gets 0
        self.play_point_impact_no_penalty = 0
        self.play_point_impact_no_opp_penalty = 0
        self.play_literature_pointvalue = 0 # see linup efficiency paper
        self.point_value = 0
        self.type = None
        self.is_shot = False
        self.by = None
        self.play_weight = None

    def convert_time(self, time):
        """converting a time in the format '11:25.0' to the number of seconds left in the quatre"""

        time_obj = datetime.strptime(time[:-2], "%M:%S")
        return time_obj.second + (time_obj.minute * 60)

    def calc_score_diff(self, score):

        score_lst = [int(i) for i in score.split("-")]

        if self.left_team == self.target_team:
            score_diff = score_lst[0] - score_lst[1]

        elif self.right_team == self.target_team:
            score_diff = score_lst[1] - score_lst[0]

        else:
            raise Exception(f"Neither the left or right team attributes are the target team ({self.target_team})")

        return score_diff


    def classify_play(self):
        if len(self.split_text) > 2: # shot
            self.is_shot = True
            if self.split_text[2] == "misses":
                self.positive = False
                self.check_shot_type()

            elif self.split_text[2] == "makes":
                # self.positive = True
                self.check_shot_type()

            elif self.split_text[1] == "rebound":
                rebound_type = self.split_text[0]
                self.type = f"{rebound_type} {self.split_text[1]}"
                self.by = self.text.split("by")[-1]
                self.play_literature_pointvalue = 1

            elif "foul" in self.split_text:
                self.type = "foul"
                self.by = " ".join(self.split_text[3:5])
                self.positive = False
                self.play_literature_pointvalue = -1

            elif self.split_text[0] == "Turnover":
                self.type = "turnover"
                self.by = self.split_text[2:4]
                self.positive = False
                self.play_literature_pointvalue = -1

            elif self.split_text[2] == "enters":
                self.type = "substitution"

            if self.team != self.target_team:
                self.positive = not self.positive

        # self.assign_weight()

    def check_shot_type(self):

        if self.split_text[4] in ["dunk", "jump", "layup"]: # "hook" taken out of this list
            pts = self.split_text[3] # point value (in the format "2-pts")
            point_value = int(pts[0])
            self.point_value = point_value
            miss_or_make = self.split_text[2] # either the string miss or make

            if miss_or_make == "misses":
                self.play_point_impact = point_value * -1
                self.play_point_impact_no_opp_penalty = point_value * -1
                self.play_literature_pointvalue = -0.5

            elif miss_or_make == "makes":
                self.play_point_impact = point_value
                self.play_point_impact_no_penalty = point_value
                self.play_point_impact_no_opp_penalty = point_value
                self.play_literature_pointvalue = point_value

            else:
                raise Exception("invalid value for miss_or_make. Bad data input?")

            if self.team != self.target_team:
                self.play_point_impact *= -1
                self.play_point_impact_no_penalty = 0
                self.play_point_impact_no_opp_penalty = 0

            shot_info = self.split_text[4] # e.g. dunk, layup, jump etc
            self.type = f"{shot_info} {'shot' if shot_info == 'jump' else ''}".strip()
            

        self.by = " ".join(self.split_text[0:2])

    def assign_weight(self):
        weight_dict = {
            "foul": -4,
            "turnover": -4,
            "Defensive rebound": 2,
            "Offensive rebound": 2,
            "misses 3-pt jump shot": -2,
            "makes 3-pt jump shot": 3,
            "makes 2-pt jump shot": 2,
            "misses 2-pt jump shot": -2,
            "makes 2-pt layup": 1,
            "misses 2-pt layup": -2,
            "makes 2-pt dunk": 2,
            "misses 2-pt dunk": -2,
            "makes 2-pt hook": 2,
            "misses 2-pt hook": -2,
            "misses free throw": -1,
            "makes free throw": 1   
        }

        if self.type != "substitution" and self.type:
            self.play_weight = weight_dict[self.type]




