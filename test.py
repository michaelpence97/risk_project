import os
import pandas as pd
import datetime
from RunningForScenario import run_scenario
import time


start_time = time.time()



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

stack_heights = (100, 800) #form is (uncontrolled_h, controlled_h)
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

elapsed_time = time.time() - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
