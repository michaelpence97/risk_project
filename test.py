import os
import pandas as pd
import datetime
from RunningForScenario import run_scenario

# Ensure the Cleaned_Data folder exists
if not os.path.exists("Cleaned_Data"):
    os.makedirs("Cleaned_Data")

# Check if cleaned data exists, otherwise run data_cleaning.py
if not os.path.isfile("Cleaned_Data/cleaned_demographic_data.xlsx"):
    os.system("python data_cleaning.py")


fcarstart = datetime.datetime(2023, 2, 6, 16, 30)
fcarstop = datetime.datetime(2023, 2, 6, 17, 0)
carstart = datetime.datetime(2023, 2, 6, 17, 0)
carstop = datetime.datetime(2023, 2, 7, 17, 0)
ethylcarstart = datetime.datetime(2023, 2, 3, 20, 45)
ethylcarstop = datetime.datetime(2023, 2, 7, 17, 0)
butylcarstart = datetime.datetime(2023, 2, 3, 20, 45)
butylcarstop = datetime.datetime(2023, 2, 7, 17, 0)

fcar = (0, 0)
car = (0, 0)
ethylcar = (0, 0)
butylcar = (0, 0)

fcar_times = (fcarstart, fcarstop)
car_times = (carstart, carstop)
ethylcar_times = (ethylcarstart, ethylcarstop)
butylcar_times = (butylcarstart, butylcarstop)

stack_heights = (100, 800)
scenario_output_folder = "Test_Scenario_Output"
run_scenario(
    fcar,
    car,
    ethylcar,
    butylcar,
    fcar_times,
    car_times,
    ethylcar_times,
    butylcar_times,
    stack_heights,
    scenario_output_folder,
)
