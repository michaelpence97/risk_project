import os
import datetime
from RunningForScenario import run_scenario
import time
import csv


start_time = time.time()

controlled_start = datetime.datetime(2023, 2, 6, 16, 30)
controlled_stop = datetime.datetime(2023, 2, 6, 17, 0)
uncontrolled_start = datetime.datetime(2023, 2, 3, 20, 45)
uncontrolled_stop = datetime.datetime(2023, 2, 7, 17, 0)

plume_x = 250
plume_origin_on_map = (plume_x, 0)
fcar = plume_origin_on_map
car = plume_origin_on_map
ethylcar = plume_origin_on_map
butylcar = plume_origin_on_map

fcar_times = (controlled_start, controlled_stop)
car_times = (controlled_start, controlled_stop)
ethylcar_times = (uncontrolled_start, uncontrolled_stop)
butylcar_times = (uncontrolled_start, uncontrolled_stop)

stack_heights = (5, 1400) #form is (uncontrolled_h, controlled_h)

def log_scenario_data(scenario_name, controlled_start, controlled_stop, plume_x, stack_heights):
    log_file = "scenario_log.csv"
    # Check if the log file exists, if not create it and add the header
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario Name", "Controlled Start", "Controlled Stop", "Plume X", "Uncontrolled Stack Height", "Controlled Stack Height"])

    # Append the scenario data to the log file
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([scenario_name, controlled_start, controlled_stop, plume_x, stack_heights[0], stack_heights[1]])


scenario_output_folder = "Scenario 12"
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


log_scenario_data(scenario_output_folder, controlled_start, controlled_stop, plume_x, stack_heights)