import os

for path in os.listdir("2021_22_MIL_Data_for_histograms"):
    # rename each file to start with 2021_22 instead fo 2014_15
    os.rename(f"2021_22_MIL_Data_for_histograms/{path}", path.replace("2021_22", "MIL_2021_22"))
