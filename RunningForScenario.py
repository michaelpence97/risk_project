import pandas as pd
from matplotlib.ticker import ScalarFormatter
import os
import glob
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from Get_concentrations_and_HQs import create_plume_locations, get_concentrations_dict, add_time_weighted_averages, create_twa_dataframes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# Load cleaned data
demographic_data = pd.read_csv('Cleaned_Data/cleaned_demographic_data.csv', index_col=0)
weather_final = pd.read_csv('Cleaned_Data/cleaned_weather_data.csv', index_col=0, parse_dates=True)
flows_and_rates = pd.read_csv('Cleaned_Data/cleaned_flows_and_rates.csv', index_col=0)
exposure_limits = pd.read_csv('Cleaned_Data/exposure_limits.csv', index_col=0)

def draw_evacuation_circle(ax, center, radius_mi, linestyle='--', edgecolor='k', linewidth=2):
    # Convert the radius from miles to meters
    radius_m = radius_mi * 1609.34
    circle = mpatches.Circle(center, radius_m, fill=False, linestyle=linestyle, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)
def generate_heatmaps(twa_dataframes, compound, demographic_data, output_folder, pac_level, center):
    evacuation_info = [
        {
            'start': datetime.strptime('2023-02-03 20:54:00', '%Y-%m-%d %H:%M:%S'),
            'end': datetime.strptime('2023-02-08 17:00:00', '%Y-%m-%d %H:%M:%S'),
            'radius': 1
        },
        {
            'start': datetime.strptime('2023-02-06 16:30:00', '%Y-%m-%d %H:%M:%S'),
            'end': datetime.strptime('2023-02-08 17:00:00', '%Y-%m-%d %H:%M:%S'),
            'radius': 2
        }
    ]
    compound_dataframes = twa_dataframes[compound]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find global minimum and maximum hazard index values
    global_min = min(df[f'{compound} Hazard Index PAC{pac_level}'].min() for df in compound_dataframes.values())
    global_max = max(df[f'{compound} Hazard Index PAC{pac_level}'].max() for df in compound_dataframes.values())

    for time_interval, twa_df in tqdm(compound_dataframes.items(), desc=f"Generating heatmaps for {compound} (PAC{pac_level})"):
        twa_df_with_coordinates = twa_df.merge(demographic_data[['NAME', 'x', 'y', 'Total Population']], left_index=True, right_on='NAME')

        # Create a scatterplot with x, y coordinates and hazard index as color scale
        plt.figure(figsize=(10, 10))
        ax = sns.scatterplot(data=twa_df_with_coordinates, x='x', y='y', hue=f'{compound} Hazard Index PAC{pac_level}', size='Total Population', sizes=(20, 200), palette='coolwarm')
        for info in evacuation_info:
            if info['start'] <= time_interval <= info['end']:
                draw_evacuation_circle(ax, center, info['radius'])

        # Add a colorbar with consistent scale
        norm = plt.Normalize(global_min, global_max)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label=f'{compound} Hazard Index PAC{pac_level}')

        # Add a legend for sizes
        legend_labels = ['Low', 'High']
        legend_points = [plt.scatter([], [], s=20, color='gray'), plt.scatter([], [], s=200, color='gray')]
        plt.legend(legend_points, legend_labels, title='Population', loc='lower right')

        # Display the timestamp
        plt.title(f"Time: {time_interval.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12, pad=12)

        timestamp_str = time_interval.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(os.path.join(output_folder, f"{compound}_heatmap_PAC{pac_level}_{timestamp_str}.png"))
        plt.close()


def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponent of base 10 and the input value.
    """
    return f"{10 ** x:.0f}"

def direct_value(x, pos):
    """The two args are the value and tick position.
    Label ticks with the input value.
    """
    return f"{x:.2f}"

def generate_cumulative_hazard_index_heatmap(demographic_data, column, output_path):
    plt.figure(figsize=(10, 10))
    scatter_plot = plt.gca()  # Get the current axes object

    scaling_factor = 100  # Adjust this value to change the size of the scatter points
    scatter_points = scatter_plot.scatter(
        demographic_data["x"],
        demographic_data["y"],
        s=demographic_data["Total Population"] / 100 * scaling_factor,
        c=demographic_data[column],
        cmap='YlOrRd',
        norm=colors.LogNorm(),
        alpha=0.5,
    )

    #add a colorbar with scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2,2))
    cbar = plt.colorbar(scatter_points, format=formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)



    scatter_plot.set_xlabel("East-West Distance from Source (m)")
    scatter_plot.set_ylabel("North-South Distance from Source (m)")

    # Draw evacuation circles with radii of 1 and 2 miles
    center = (0, 0)  # Assuming the center is at (0, 0), adjust this if necessary
    draw_evacuation_circle(scatter_plot, center, 1)
    draw_evacuation_circle(scatter_plot, center, 2)

    plt.savefig(output_path)
    plt.close()

def generate_gif_from_heatmaps(heatmaps_folder, output_path, twa_dataframes, compound):
    images = []
    durations = [900]
    heatmap_files = sorted(glob.glob(os.path.join(heatmaps_folder, f"{compound}_heatmap_*.png")))

    # Get the time intervals between images and calculate the durations for the gif frames
    time_intervals = list(twa_dataframes[compound].keys())
    for i in range(len(time_intervals) - 1):
        time_diff_seconds = int((time_intervals[i + 1] - time_intervals[i]).total_seconds())
        durations.append(time_diff_seconds)

    if len(durations) != len(heatmap_files) - 1:
        print(f"Error: Mismatch in the lengths of durations ({len(durations)}) and heatmap_files ({len(heatmap_files)}).")

    # Load heatmap images and append to the images list
    for file in heatmap_files:
        images.append(Image.open(file))

    # Calculate the scaling factor
    total_desired_playback_time = 30  # 15 seconds
    scaling_factor = total_desired_playback_time / sum(durations)

    # Scale the durations
    scaled_durations = [int(duration * scaling_factor * 100) for duration in durations]  # Convert to milliseconds

    print(f"Number of images: {len(images)}, Number of durations: {len(durations)}")

    # Create a gif from the images with the specified durations
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=scaled_durations, loop=0)


def run_scenario(fcar, car, ethylcar, butylcar, fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights, scenario_output_folder):
    # Create plume_locations and get concentrations_dict
    plume_locations = create_plume_locations(fcar, car, ethylcar, butylcar, fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights)
    concentrations_dict = get_concentrations_dict(demographic_data, weather_final, flows_and_rates, plume_locations)
    concentrations_dict = add_time_weighted_averages(concentrations_dict)
    # Fill NaN values with 0 in TWA columns
    for name, concentration_data in concentrations_dict.items():
        twa_columns = [col for col in concentration_data.columns if 'TWA' in col]
        concentration_data[twa_columns] = concentration_data[twa_columns].fillna(0)
    twa_dataframes = create_twa_dataframes(concentrations_dict, exposure_limits)

    concentrations_csv_folder = os.path.join(scenario_output_folder, "concentrations_csv")
    if not os.path.exists(concentrations_csv_folder):
        os.makedirs(concentrations_csv_folder)

    for name, concentration_df in concentrations_dict.items():
        concentration_df.to_csv(os.path.join(concentrations_csv_folder, f"{name}_concentration.csv"))

    # Save twa_dataframes dataframes as CSVs
    twa_csv_folder = os.path.join(scenario_output_folder, "twa_csv")
    if not os.path.exists(twa_csv_folder):
        os.makedirs(twa_csv_folder)

    for compound, compound_dataframes in twa_dataframes.items():
        compound_folder = os.path.join(twa_csv_folder, compound)
        if not os.path.exists(compound_folder):
            os.makedirs(compound_folder)

        for time_interval, twa_df in compound_dataframes.items():
            timestamp_str = time_interval.strftime("%Y-%m-%d_%H-%M-%S")
            twa_df.to_csv(os.path.join(compound_folder, f"{compound}_twa_{timestamp_str}.csv"))

    # Initialize the cumulative hazard index dictionary
    cumulative_hazard_index = {}

    for compound in twa_dataframes.keys():
        compound_dataframes = twa_dataframes[compound]
        cumulative_hazard_index_compound = {}

        for pac in ['PAC1', 'PAC2', 'PAC3']:
            cumulative_hazard_index_pac = pd.DataFrame(index=demographic_data['NAME'].unique())

            previous_time_interval = None

            for time_interval, twa_df in compound_dataframes.items():
                # Calculate the duration of the time interval in hours
                if previous_time_interval is not None:
                    duration_seconds = (time_interval - previous_time_interval).total_seconds()
                    if compound in ["PM2.5", "PM10"]:
                        duration = duration_seconds / (24 * 3600)
                    else:
                        duration = duration_seconds / 3600
                else:
                    duration = .25 if compound not in ["PM2.5", "PM10"] else .25/24 # Assuming 15 min for the first interval

                # Multiply the hazard index by the duration and add it to the cumulative hazard index
                hazard_index_column = f"{compound} Hazard Index {pac}"
                cumulative_hazard_index_pac = cumulative_hazard_index_pac.fillna(0)
                hazard_index_duration = twa_df[hazard_index_column] * duration
                hazard_index_duration = hazard_index_duration.fillna(0)

                # Fix: Create a new column with the appropriate name and update it with hazard_index_duration values
                column_name = f"{compound} Cumulative Hazard Index {pac}"
                cumulative_hazard_index_pac[column_name] = cumulative_hazard_index_pac.get(column_name,
                                                                                           0) + hazard_index_duration

                previous_time_interval = time_interval

            cumulative_hazard_index_compound[pac] = cumulative_hazard_index_pac

        cumulative_hazard_index[compound] = cumulative_hazard_index_compound

        # Rest of the run_scenario code
        # Create a new DataFrame as a copy of demographic_data
    demographic_data_with_cumulative_hazard_index = demographic_data.copy()

    # Add the cumulative hazard index values as new columns
    for compound, compound_hazard_index in cumulative_hazard_index.items():
        for pac, pac_hazard_index in compound_hazard_index.items():
            column_name = f"{compound} Cumulative Hazard Index {pac}"
            pac_hazard_index = pac_hazard_index.reset_index().rename(columns={"index": "NAME", 0: column_name})
            demographic_data_with_cumulative_hazard_index = demographic_data_with_cumulative_hazard_index.merge(
                pac_hazard_index, on="NAME", how="left")

    #cancers
    additional_cancer_prob = {}
    previous_time_interval = None
    inhalation_unit_risk = 8.8*(10**(-6))
    for time_interval, twa_df in twa_dataframes["Total Vinyl Chloride"].items():
        if previous_time_interval is not None:
            duration_seconds = (time_interval - previous_time_interval).total_seconds()
            duration = duration_seconds / 3600
            twa_df = twa_df.copy()
            twa_df['Additional Cancer Prob'] = twa_df[
                                                   "Total Vinyl Chloride TWA 60 min"] * duration * inhalation_unit_risk
            additional_cancer_prob[time_interval] = twa_df
        previous_time_interval = time_interval
    total_cancer_prob = pd.DataFrame(index=demographic_data['NAME'].unique())
    for time_interval, cancer_prob_df in additional_cancer_prob.items():
        total_cancer_prob = total_cancer_prob.add(cancer_prob_df[['Additional Cancer Prob']], axis='index').fillna(0)

    demographic_data_with_cancer_prob = demographic_data.merge(total_cancer_prob, left_on='NAME', right_index=True,
                                                               how='left')
    demographic_data_with_cancer_prob["Expected Cancers"] = demographic_data_with_cancer_prob["Total Population"] * \
                                                            demographic_data_with_cancer_prob["Additional Cancer Prob"]

    # Save demographic data with cancer probability and expected cancers as a CSV file
    demographic_cancer_prob_csv_path = os.path.join(scenario_output_folder, "demographic_data_with_cancer_prob.csv")
    demographic_data_with_cancer_prob.to_csv(demographic_cancer_prob_csv_path)

    total_expected_cancers = demographic_data_with_cancer_prob["Expected Cancers"].sum()

    # Create a DataFrame with the total expected cancers value
    total_expected_cancers_df = pd.DataFrame(data={"Total Expected Cancers": [total_expected_cancers]})
    # Save the total expected cancers as a CSV file
    total_expected_cancers_csv_path = os.path.join(scenario_output_folder, "total_expected_cancers.csv")
    total_expected_cancers_df.to_csv(total_expected_cancers_csv_path, index=False)

    # Initialize the community-level hazard index dictionary
    community_level_hazard_index = {}

    for compound in twa_dataframes.keys():
        compound_hazard_index = cumulative_hazard_index[compound]
        community_level_hazard_index_compound = {}

        for pac, pac_hazard_index in compound_hazard_index.items():
            # Reset the index for both dataframes
            pac_hazard_index = pac_hazard_index.reset_index(drop=True)
            demographic_data_with_cumulative_hazard_index = demographic_data_with_cumulative_hazard_index.reset_index(
                drop=True)

            # Calculate the sum of the product of the cumulative hazard index and population for each census block group
            product = []
            for index, row in pac_hazard_index.iterrows():
                pop = demographic_data_with_cumulative_hazard_index.loc[index, 'Total Population']
                pac_hazard_value = row[f"{compound} Cumulative Hazard Index {pac}"]
                product.append(pop * pac_hazard_value)
            product = pd.Series(product, index=pac_hazard_index.index)
            community_level_hazard_index_pac = product.sum()
            community_level_hazard_index_compound[pac] = community_level_hazard_index_pac

        community_level_hazard_index[compound] = community_level_hazard_index_compound

    # Save cumulative hazard index data as CSV files
    cumulative_hazard_index_folder = os.path.join(scenario_output_folder, "cumulative_hazard_index_csv")
    if not os.path.exists(cumulative_hazard_index_folder):
        os.makedirs(cumulative_hazard_index_folder)

    for compound, compound_hazard_index in cumulative_hazard_index.items():
        for pac, pac_hazard_index in compound_hazard_index.items():
            pac_hazard_index_csv_path = os.path.join(cumulative_hazard_index_folder, f"{compound}_Cumulative_Hazard_Index_{pac}.csv")
            pac_hazard_index.to_csv(pac_hazard_index_csv_path)

    # Save community-level hazard index data as a CSV file
    community_level_hazard_index_df = pd.DataFrame.from_dict(community_level_hazard_index, orient="index")
    community_level_hazard_index_df.columns = [f"Community Level Hazard Index {pac}" for pac in
                                               community_level_hazard_index_df.columns]
    community_level_hazard_index_csv_path = os.path.join(scenario_output_folder, "community_level_hazard_index.csv")
    community_level_hazard_index_df.to_csv(community_level_hazard_index_csv_path)

    heatmap_output_folder = os.path.join(scenario_output_folder, "cumulative_hazard_index_heatmaps")
    if not os.path.exists(heatmap_output_folder):
        os.makedirs(heatmap_output_folder)

    for compound in twa_dataframes.keys():
        for pac in ['PAC1', 'PAC2', 'PAC3']:
            column_name = f"{compound} Cumulative Hazard Index {pac}"
            output_path = os.path.join(heatmap_output_folder, f"{compound}_Cumulative_Hazard_Index_{pac}.png")
            generate_cumulative_hazard_index_heatmap(demographic_data_with_cumulative_hazard_index, column_name,
                                                     output_path)


def run_scenario2(fcar, car, ethylcar, butylcar, fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights, scenario_output_folder):
    # Create plume_locations and get concentrations_dict
    plume_locations = create_plume_locations(fcar, car, ethylcar, butylcar, fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights)
    concentrations_dict = get_concentrations_dict(demographic_data, weather_final, flows_and_rates, plume_locations)
    concentrations_dict = add_time_weighted_averages(concentrations_dict)
    # Fill NaN values with 0 in TWA columns
    for name, concentration_data in concentrations_dict.items():
        twa_columns = [col for col in concentration_data.columns if 'TWA' in col]
        concentration_data[twa_columns] = concentration_data[twa_columns].fillna(0)
    twa_dataframes = create_twa_dataframes(concentrations_dict, exposure_limits)

    concentrations_csv_folder = os.path.join(scenario_output_folder, "concentrations_csv")
    if not os.path.exists(concentrations_csv_folder):
        os.makedirs(concentrations_csv_folder)

    for name, concentration_df in concentrations_dict.items():
        concentration_df.to_csv(os.path.join(concentrations_csv_folder, f"{name}_concentration.csv"))

    # Save twa_dataframes dataframes as CSVs
    twa_csv_folder = os.path.join(scenario_output_folder, "twa_csv")
    if not os.path.exists(twa_csv_folder):
        os.makedirs(twa_csv_folder)

    for compound, compound_dataframes in twa_dataframes.items():
        compound_folder = os.path.join(twa_csv_folder, compound)
        if not os.path.exists(compound_folder):
            os.makedirs(compound_folder)

        for time_interval, twa_df in compound_dataframes.items():
            timestamp_str = time_interval.strftime("%Y-%m-%d_%H-%M-%S")
            twa_df.to_csv(os.path.join(compound_folder, f"{compound}_twa_{timestamp_str}.csv"))

    # Initialize the cumulative hazard index dictionary
    cumulative_hazard_index = {}

    for compound in twa_dataframes.keys():
        compound_dataframes = twa_dataframes[compound]
        cumulative_hazard_index_compound = {}

        for pac in ['PAC1', 'PAC2', 'PAC3']:
            cumulative_hazard_index_pac = pd.DataFrame(index=demographic_data['NAME'].unique())

            previous_time_interval = None

            for time_interval, twa_df in compound_dataframes.items():
                # Calculate the duration of the time interval in hours
                if previous_time_interval is not None:
                    duration_seconds = (time_interval - previous_time_interval).total_seconds()
                    if compound in ["PM2.5", "PM10"]:
                        duration = duration_seconds / (24 * 3600)
                    else:
                        duration = duration_seconds / 3600
                else:
                    duration = .25 if compound not in ["PM2.5", "PM10"] else .25/24 # Assuming 15 min for the first interval

                # Multiply the hazard index by the duration and add it to the cumulative hazard index
                hazard_index_column = f"{compound} Hazard Index {pac}"
                cumulative_hazard_index_pac = cumulative_hazard_index_pac.fillna(0)
                hazard_index_duration = twa_df[hazard_index_column] * duration
                hazard_index_duration = hazard_index_duration.fillna(0)

                # Fix: Create a new column with the appropriate name and update it with hazard_index_duration values
                column_name = f"{compound} Cumulative Hazard Index {pac}"
                cumulative_hazard_index_pac[column_name] = cumulative_hazard_index_pac.get(column_name,
                                                                                           0) + hazard_index_duration

                previous_time_interval = time_interval

            cumulative_hazard_index_compound[pac] = cumulative_hazard_index_pac

        cumulative_hazard_index[compound] = cumulative_hazard_index_compound

        # Rest of the run_scenario code
        # Create a new DataFrame as a copy of demographic_data
    demographic_data_with_cumulative_hazard_index = demographic_data.copy()

    # Add the cumulative hazard index values as new columns
    for compound, compound_hazard_index in cumulative_hazard_index.items():
        for pac, pac_hazard_index in compound_hazard_index.items():
            column_name = f"{compound} Cumulative Hazard Index {pac}"
            pac_hazard_index = pac_hazard_index.reset_index().rename(columns={"index": "NAME", 0: column_name})
            demographic_data_with_cumulative_hazard_index = demographic_data_with_cumulative_hazard_index.merge(
                pac_hazard_index, on="NAME", how="left")

    #additional probability of developing cancer:
    inhalation_unit_risk = 8.8e-06
    additional_cancer_prob = {}
    for time_interval, twa_df in twa_dataframes["Total Vinyl Chloride"].items():
        duration_seconds = (time_interval - previous_time_interval).total_seconds() if previous_time_interval else 3600*0.25
        duration = duration_seconds / 3600
        additional_cancer_prob[time_interval] = twa_df["Total Vinyl Chloride TWA 60 min"]*duration*inhalation_unit_risk
    total_cancer_prob = pd.DataFrame(index=demographic_data['NAME'].unique())
    for time_interval, cancer_prob_df in additional_cancer_prob.items():
        total_cancer_prob = total_cancer_prob.add(cancer_prob_df).fillna(0)

    demographic_data_with_cancer_prob = demographic_data.copy().merge(total_cancer_prob, left_on='NAME', right_index=True, how='left')
    demographic_data_with_cancer_prob["Expected Cancers"] = demographic_data_with_cancer_prob["Total Population"]*demographic_data_with_cancer_prob["Total Vinyl Chloride TWA 60 min"]
    total_expected_cancers = demographic_data_with_cancer_prob["Expected Cancers"].sum()

    # Save demographic data with cancer probability and expected cancers as a CSV file
    demographic_cancer_prob_csv_path = os.path.join(scenario_output_folder, "demographic_data_with_cancer_prob.csv")
    demographic_data_with_cancer_prob.to_csv(demographic_cancer_prob_csv_path)

    # Create a DataFrame with the total expected cancers value
    total_expected_cancers_df = pd.DataFrame(data={"Total Expected Cancers": [total_expected_cancers]})
    # Save the total expected cancers as a CSV file
    total_expected_cancers_csv_path = os.path.join(scenario_output_folder, "total_expected_cancers.csv")
    total_expected_cancers_df.to_csv(total_expected_cancers_csv_path, index=False)

    # Initialize the community-level hazard index dictionary
    community_level_hazard_index = {}

    for compound in twa_dataframes.keys():
        compound_hazard_index = cumulative_hazard_index[compound]
        community_level_hazard_index_compound = {}

        for pac, pac_hazard_index in compound_hazard_index.items():
            # Calculate the sum of the product of the cumulative hazard index and population for each census block group
            product = demographic_data_with_cumulative_hazard_index['Total Population'].multiply(
                pac_hazard_index.loc[:, f"{compound} Cumulative Hazard Index {pac}"], fill_value=0)

            # Debugging print statements
            print(f"DEBUG: {compound}, {pac}")
            print("Total Population:")
            print(demographic_data_with_cumulative_hazard_index['Total Population'].head())
            print("PAC Hazard Index:")
            print(pac_hazard_index.loc[:, f"{compound} Cumulative Hazard Index {pac}"].head())
            print("Product:")
            print(product.head())
            community_level_hazard_index_pac = product.sum()
            community_level_hazard_index_compound[pac] = community_level_hazard_index_pac

        community_level_hazard_index[compound] = community_level_hazard_index_compound

    # Save cumulative hazard index data as CSV files
    cumulative_hazard_index_folder = os.path.join(scenario_output_folder, "cumulative_hazard_index_csv")
    if not os.path.exists(cumulative_hazard_index_folder):
        os.makedirs(cumulative_hazard_index_folder)

    for compound, compound_hazard_index in cumulative_hazard_index.items():
        for pac, pac_hazard_index in compound_hazard_index.items():
            pac_hazard_index_csv_path = os.path.join(cumulative_hazard_index_folder, f"{compound}_Cumulative_Hazard_Index_{pac}.csv")
            pac_hazard_index.to_csv(pac_hazard_index_csv_path)

    # Save community-level hazard index data as a CSV file
    community_level_hazard_index_df = pd.DataFrame.from_dict(community_level_hazard_index, orient="index")
    community_level_hazard_index_df.columns = [f"Community Level Hazard Index {pac}" for pac in
                                               community_level_hazard_index_df.columns]
    community_level_hazard_index_csv_path = os.path.join(scenario_output_folder, "community_level_hazard_index.csv")
    community_level_hazard_index_df.to_csv(community_level_hazard_index_csv_path)

    heatmap_output_folder = os.path.join(scenario_output_folder, "cumulative_hazard_index_heatmaps")
    if not os.path.exists(heatmap_output_folder):
        os.makedirs(heatmap_output_folder)

    for compound in twa_dataframes.keys():
        for pac in ['PAC1', 'PAC2', 'PAC3']:
            column_name = f"{compound} Cumulative Hazard Index {pac}"
            output_path = os.path.join(heatmap_output_folder, f"{compound}_Cumulative_Hazard_Index_{pac}.png")
            generate_cumulative_hazard_index_heatmap(demographic_data_with_cumulative_hazard_index, column_name,
                                                     output_path)







def run_scenario1(fcar, car, ethylcar, butylcar, fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights, scenario_output_folder):
    # Create plume_locations and get concentrations_dict
    plume_locations = create_plume_locations(fcar, car, ethylcar, butylcar, fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights)
    concentrations_dict = get_concentrations_dict(demographic_data, weather_final, flows_and_rates, plume_locations)
    concentrations_dict = add_time_weighted_averages(concentrations_dict)
    # Fill NaN values with 0 in TWA columns
    for name, concentration_data in concentrations_dict.items():
        twa_columns = [col for col in concentration_data.columns if 'TWA' in col]
        concentration_data[twa_columns] = concentration_data[twa_columns].fillna(0)
    twa_dataframes = create_twa_dataframes(concentrations_dict, exposure_limits)

    concentrations_csv_folder = os.path.join(scenario_output_folder, "concentrations_csv")
    if not os.path.exists(concentrations_csv_folder):
        os.makedirs(concentrations_csv_folder)

    for name, concentration_df in concentrations_dict.items():
        concentration_df.to_csv(os.path.join(concentrations_csv_folder, f"{name}_concentration.csv"))

    # Save twa_dataframes dataframes as CSVs
    twa_csv_folder = os.path.join(scenario_output_folder, "twa_csv")
    if not os.path.exists(twa_csv_folder):
        os.makedirs(twa_csv_folder)

    for compound, compound_dataframes in twa_dataframes.items():
        compound_folder = os.path.join(twa_csv_folder, compound)
        if not os.path.exists(compound_folder):
            os.makedirs(compound_folder)

        for time_interval, twa_df in compound_dataframes.items():
            timestamp_str = time_interval.strftime("%Y-%m-%d_%H-%M-%S")
            twa_df.to_csv(os.path.join(compound_folder, f"{compound}_twa_{timestamp_str}.csv"))

    # Generate heatmaps
    for compound in twa_dataframes.keys():
        for pac_level in [1, 2, 3]:
            heatmap_output_folder = os.path.join(scenario_output_folder, f"{compound}_heatmaps_PAC{pac_level}")
            generate_heatmaps(twa_dataframes, compound, demographic_data, heatmap_output_folder, pac_level, center=(0, 0))

            # Generate a gif from the heatmaps
            gif_output_path = os.path.join(heatmap_output_folder, f"{compound}_heatmaps_PAC{pac_level}.gif")
            generate_gif_from_heatmaps(heatmap_output_folder, gif_output_path, twa_dataframes, compound)

    # Initialize the cumulative hazard index dictionary
    cumulative_hazard_index = {}

    for compound in twa_dataframes.keys():
        compound_dataframes = twa_dataframes[compound]
        cumulative_hazard_index_compound = {}

        for pac in ['PAC1', 'PAC2', 'PAC3']:
            cumulative_hazard_index_pac = pd.DataFrame(index=demographic_data['NAME'].unique())

            previous_time_interval = None

            for time_interval, twa_df in compound_dataframes.items():
                # Calculate the duration of the time interval in hours
                if previous_time_interval is not None:
                    duration = (time_interval - previous_time_interval).total_seconds() / 3600
                else:
                    duration = 1  # Assuming 1 hour for the first interval

                # Multiply the hazard index by the duration and add it to the cumulative hazard index
                hazard_index_column = f"{compound} Hazard Index {pac}"
                cumulative_hazard_index_pac = cumulative_hazard_index_pac.add(
                    twa_df[hazard_index_column] * duration).fillna(0)

                previous_time_interval = time_interval

            cumulative_hazard_index_compound[pac] = cumulative_hazard_index_pac

        cumulative_hazard_index[compound] = cumulative_hazard_index_compound

        # Rest of the run_scenario code
        # Create a new DataFrame as a copy of demographic_data
    demographic_data_with_cumulative_hazard_index = demographic_data.copy()

    # Add the cumulative hazard index values as new columns
    for compound, compound_hazard_index in cumulative_hazard_index.items():
        for pac, pac_hazard_index in compound_hazard_index.items():
            column_name = f"{compound} Cumulative Hazard Index {pac}"
            demographic_data_with_cumulative_hazard_index[column_name] = demographic_data_with_cumulative_hazard_index[
                'NAME'].map(pac_hazard_index)

    # Initialize the community-level hazard index dictionary
    community_level_hazard_index = {}

    for compound in twa_dataframes.keys():
        compound_hazard_index = cumulative_hazard_index[compound]
        community_level_hazard_index_compound = {}

        for pac, pac_hazard_index in compound_hazard_index.items():
            community_level_hazard_index_compound[pac] = (demographic_data['Total Population'] * pac_hazard_index).sum()

        community_level_hazard_index[compound] = community_level_hazard_index_compound

    # Save cumulative hazard index data as CSV files
    cumulative_hazard_index_folder = os.path.join(scenario_output_folder, "cumulative_hazard_index_csv")
    if not os.path.exists(cumulative_hazard_index_folder):
        os.makedirs(cumulative_hazard_index_folder)

    for compound, compound_hazard_index in cumulative_hazard_index.items():
        for pac, pac_hazard_index in compound_hazard_index.items():
            pac_hazard_index_csv_path = os.path.join(cumulative_hazard_index_folder, f"{compound}_Cumulative_Hazard_Index_{pac}.csv")
            pac_hazard_index.to_csv(pac_hazard_index_csv_path)

    # Save community-level hazard index data as a CSV file
    community_level_hazard_index_df = pd.DataFrame(community_level_hazard_index)
    community_level_hazard_index_csv_path = os.path.join(scenario_output_folder, "community_level_hazard_index.csv")
    community_level_hazard_index_df.to_csv(community_level_hazard_index_csv_path)

    heatmap_output_folder = os.path.join(scenario_output_folder, "cumulative_hazard_index_heatmaps")
    if not os.path.exists(heatmap_output_folder):
        os.makedirs(heatmap_output_folder)

    for compound in twa_dataframes.keys():
        for pac in ['PAC1', 'PAC2', 'PAC3']:
            column_name = f"{compound} Cumulative Hazard Index {pac}"
            output_path = os.path.join(heatmap_output_folder, f"{compound}_Cumulative_Hazard_Index_{pac}.png")
            generate_cumulative_hazard_index_heatmap(demographic_data_with_cumulative_hazard_index, column_name,
                                                     output_path)

