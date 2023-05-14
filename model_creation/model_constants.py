import time

class ModelConstants:
    EPOCHS = 70
    BATCH_SIZE = 32
    SEQ_LEN = 10
    FUTURE_PERIOD_PREDICT = 10
    # TEAMS_USED = ["MIL"]
    FEATURE_INFO = "-".join(["GOODBAD", "SCOREDIFF", "TIME"])
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{FEATURE_INFO}-FEATUREINFO-{time.time()}"
    CSV_COLS = [
    "game number",
    "time",
    "score_diff",
    "team",
    "good_or_bad",
    "current_sub_status",
    "play_Defensive rebound",
    "play_Offensive rebound",
    "play_foul",
    "play_makes 2-pt dunk",
    "play_makes 2-pt jump shot",
    "play_makes 2-pt layup",
    "play_makes 3-pt jump shot",
    "play_misses 2-pt dunk",
    "play_misses 2-pt jump shot",
    "play_misses 2-pt layup",
    "play_misses 3-pt jump shot",
    "play_substitution",
    "play_turnover",
    "future_sub_status",
    ]
    FEATURES = [
    "time_seconds",
    "score_diff",
    "team",
    "good_or_bad",
    "play_Defensive rebound",
    "play_Offensive rebound",
    "play_foul",
    "play_makes 2-pt dunk",
    "play_makes 2-pt jump shot",
    "play_makes 2-pt layup",
    "play_makes 3-pt jump shot",
    "play_misses 2-pt dunk",
    "play_misses 2-pt jump shot",
    "play_misses 2-pt layup",
    "play_misses 3-pt jump shot",
    "play_turnover",
    "future_sub_status", # label
    ]