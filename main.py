import os
import sys
import subprocess
import datetime
from RunningForScenario import run_scenario
from concurrent.futures import ThreadPoolExecutor

# Check if cleaned data exists, if not, run data_cleaning.py
cleaned_data_folder = "Cleaned_Data"
cleaned_data_files = [
    "cleaned_demographic_data.xlsx",
    "cleaned_weather_data.xlsx",
    "cleaned_flows_and_rates.xlsx",
    "exposure_limits.xlsx",
]

all_cleaned_data_exists = all(
    [os.path.exists(os.path.join(cleaned_data_folder, file)) for file in cleaned_data_files]
)

if not all_cleaned_data_exists:
    print("Cleaned data not found. Running data_cleaning.py...")
    subprocess.run([sys.executable, "data_cleaning.py"])

# Sensitivity analysis variables
stack_heights_sets = [(5, 800), (5, 1100), (5, 1400), (50, 800), (50, 1100), (50, 1400), (100, 800), (100, 1100), (100, 1400)]
plume_coordinates_sets = [((0, 0), (0, 0), (0, 0), (0, 0)), ((-100, 0), (-100, 0), (-100, 0), (-100, 0)), ((100, 0), (100, 0), (100, 0), (100, 0))]
uncontrolled_start = datetime.datetime(2023, 2, 3, 20, 45)
uncontrolled_end = datetime.datetime(2023, 2, 7, 17, 0)
controlled_start_times = [datetime.datetime(2023, 2, 6, 15, 0), datetime.datetime(2023, 2, 6, 16, 0), datetime.datetime(2023, 2, 6, 17, 0), datetime.datetime(2023, 2, 6, 18, 0)]

# Initialize a counter for unique scenario output folder names
scenario_counter = 0

# Function to run a single scenario
def run_single_scenario(stack_heights, plume_coordinates, controlled_start):
    global scenario_counter
    controlled_end = controlled_start + datetime.timedelta(minutes=30)
    scenario_output_folder = f"output/scenario_{scenario_counter:03d}"  # Unique identifier for the scenario
    scenario_counter += 1

    # Create the output folder if it does not exist
    os.makedirs(scenario_output_folder, exist_ok=True)

    run_scenario(
        *plume_coordinates,
        [controlled_start, controlled_end],
        [controlled_start, controlled_end],
        [uncontrolled_start, uncontrolled_end],
        [uncontrolled_start, uncontrolled_end],
        stack_heights,
        scenario_output_folder,
    )
    return scenario_output_folder


# Loop through all combinations of sensitivity analysis variables and run scenarios in parallel
os.makedirs("output", exist_ok=True)
scenario_output_folders = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for stack_heights in stack_heights_sets:
        for plume_coordinates in plume_coordinates_sets:
            for controlled_start in controlled_start_times:
                futures.append(executor.submit(run_single_scenario, stack_heights, plume_coordinates, controlled_start))

    for future in futures:
        scenario_output_folders.append(future.result())
