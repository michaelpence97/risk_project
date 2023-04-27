import numpy as np
import pandas as pd
from math import radians, sin, cos, atan2, sqrt
import os
import random

solar_data = pd.read_excel('Initial_Data/EPSolar (1) (1).xlsx')
flows_and_rates = pd.read_excel('Initial_Data/flows and decomposition rates.xlsx')
weather_data = pd.read_excel('Initial_Data/BVI_METAR_data.xlsx')
demographic_data = pd.read_excel('Initial_Data/Six Mile Demographic Blocks1.xlsx')

# Convert the 'Q' values from kg/h to kg/s by dividing them by 3600
flows_and_rates['Q (kg/s)'] = flows_and_rates['Q '] / 3600

# Convert the 'k' values from per hour to per second by dividing them by 3600
flows_and_rates['k (per sec)'] = flows_and_rates['k (per hour)'] / 3600

# Remove 'description' and 'unit' columns from the flows_and_rates DataFrame
flows_and_rates = flows_and_rates.drop(columns=['description', 'unit', 'Q ', 'k (per hour)'])

# Step 1: Add a new "settling" column with a default value of 0 for existing plumes in flows_and_rates
flows_and_rates["settling"] = 0

mask = flows_and_rates['plume name'].str.contains('dioxin')
flows_and_rates.loc[mask, 'settling'] = 0.005



# Step 2: Define the values for PM2.5 and PM10
pm25_plume_name = "PM2.5"
pm25_release_rate = 1.033 * 10 ** (-12)  # Set PM2.5 release rate
pm25_decay_rate = 0  # Set PM2.5 decay rate
pm25settling = .0002 #m/s

pm10_plume_name = "PM10"
pm10_release_rate = 2.06667 * 10 ** (-12)  # Set PM10 release rate
pm10_decay_rate = 0  # Set PM10 decay rate
pm10settling = .005 #m/s

# Create a new DataFrame for PM2.5 and PM10
pm_data = pd.DataFrame({
    "plume name": [pm25_plume_name, pm10_plume_name],
    "Q (kg/s)": [pm25_release_rate, pm10_release_rate],
    "k (per sec)": [pm25_decay_rate, pm10_decay_rate],
    "settling": [pm25settling, pm10settling]
})

# Append the new DataFrame to the original flows_and_rates DataFrame
flows_and_rates = flows_and_rates.append(pm_data, ignore_index=True)


# Define the map origin
origin_lat = 40.8360864
origin_lon = -80.5215884

# Define a function to calculate the distance in meters between two GPS coordinates
def distance(lat1, lon1, lat2, lon2):
    R = 6373.0 # approximate radius of earth in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance_km = R * c
    distance_m = distance_km * 1000

    return distance_m

# Add two columns to the demographic_data DataFrame for x and y coordinates relative to the origin
# y-axis corresponds to latitude
demographic_data['y'] = demographic_data.apply(lambda row: distance(origin_lat, origin_lon, row['Lattitude'], origin_lon) * (1 if row['Lattitude'] >= origin_lat else -1), axis=1)
# x-axis corresponds to longitude
demographic_data['x'] = demographic_data.apply(lambda row: distance(origin_lat, origin_lon, origin_lat, row['Longitude']) * (1 if row['Longitude'] >= origin_lon else -1), axis=1)

# Remove unnecessary columns from solar_data DataFrame
solar_data = solar_data.drop(columns=['Azimuth angle', 'Longitude', 'Latitude'])

# Calculate wind direction in radians and correct it. Add 180 to go from direction wind coming from to direction wind
#going to, subtract from 90 to change from clockwise from North to counterclockwise from East. % 360 normalizes.
wind_dir = np.deg2rad(90 - (weather_data['drct'] + 180) % 360)
# Convert wind speed from knots to m/s
wind_speed = weather_data['sknt'] * 0.514444

# Add new columns for wind speed and wind direction to weather_data DataFrame
weather_data['wind_speed (m/s)'] = wind_speed
weather_data['wind_direction (deg)'] = np.rad2deg(wind_dir)

# Correct negative wind direction values
weather_data.loc[weather_data['wind_direction (deg)'] < 0, 'wind_direction (deg)'] += 360

# Define cloud cover and cloud height columns
cloud_cover_columns = ['skyc1', 'skyc2', 'skyc3']
cloud_height_columns = ['skyl1', 'skyl2', 'skyl3']

# Replace missing data with NaN in cloud cover and cloud height columns
weather_data[cloud_cover_columns] = weather_data[cloud_cover_columns].replace('M', np.nan)
weather_data[cloud_height_columns] = weather_data[cloud_height_columns].replace('M', np.nan)

# Convert cloud height columns to numeric data type
weather_data[cloud_height_columns] = weather_data[cloud_height_columns].apply(pd.to_numeric, errors='coerce')

# Define cloud cover codes for estimating mixing height and overall cloud cover
cloud_cover_codes = {
    'CLR': 0,
    'SCT': 33,
    'BKN': 67,
    'OVC': 100
}


# Function to estimate mixing height based on cloud cover and cloud height columns
def estimate_mixing_height(row):
    min_bkn_height = None
    max_height = None

    # Iterate through cloud cover and cloud height columns
    for cover_col, height_col in zip(cloud_cover_columns, cloud_height_columns):
        cover_value = row[cover_col]
        height_value = row[height_col]

        # Check if cloud cover and cloud height values are not null
        if pd.notnull(cover_value) and pd.notnull(height_value):
            cover_percentage = cloud_cover_codes.get(cover_value, 0)

            # Update min_bkn_height and max_height based on cover_percentage and height_value
            if cover_percentage >= 75:
                if min_bkn_height is None or height_value < min_bkn_height:
                    min_bkn_height = height_value

            if max_height is None or height_value > max_height:
                max_height = height_value

    # Return the estimated mixing height
    return min_bkn_height if min_bkn_height is not None else (max_height if max_height is not None else 20000)


# Calculate mixing height for each row in weather_data DataFrame
weather_data['mixing_height'] = weather_data.apply(estimate_mixing_height, axis=1)


# Function to calculate overall cloud cover based on cloud cover columns
def overall_cloud_cover(row):
    max_cloud_cover = 0

    # Iterate through cloud cover columns
    for cover_col in cloud_cover_columns:
        cover_value = row[cover_col]

        # Check if cloud cover value is not null
        if pd.notnull(cover_value):
            cover_percentage = cloud_cover_codes.get(cover_value, 0)
            max_cloud_cover = max(max_cloud_cover, cover_percentage)

    return max_cloud_cover


# Calculate overall cloud cover for each row in weather_data DataFrame
weather_data['overall_cloud_cover'] = weather_data.apply(overall_cloud_cover, axis=1)


# Define columns to drop from weather_data
columns_to_drop = ['IN', 'offset', 'NRI', 'time (UTC)', 'time (EST)', 'stab class', 'sknt', 'skyc1', 'skyc2',
                   'skyc3', 'skyl1', 'skyl2', 'skyl3', 'drct']

# Find the exact name of the 'inc ang' column
inc_ang_col_name = [col for col in weather_data.columns if 'inc ang' in col][0]

# Drop the column using the exact name
weather_data = weather_data.drop(columns=[inc_ang_col_name])

# Drop the specified columns
weather_data = weather_data.drop(columns=columns_to_drop)

# Convert the 'time' column to a datetime object
weather_data['datetime'] = pd.to_datetime(weather_data['date'].astype(str) + ' ' + weather_data['time'].astype(str))
weather_data = weather_data.drop(columns=['date', 'time'])

# Step 1: Round up the 'datetime' column to the next hour if the minutes are 55
weather_data.loc[weather_data['datetime'].dt.minute == 55, 'datetime'] += pd.Timedelta(hours=1)
weather_data.loc[weather_data['datetime'].dt.minute == 55, 'datetime'] = weather_data['datetime'].dt.floor('H')

# Step 2: Create a new DataFrame with 15-minute intervals and merge it with weather_data
# First, set the 'datetime' column as the index
weather_data = weather_data.set_index('datetime')

# Create a new DataFrame with 15-minute intervals
start_date = weather_data.index.min()
end_date = weather_data.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='15min')
new_df = pd.DataFrame(date_range, columns=['datetime']).set_index('datetime')

# Merge weather_data with the new DataFrame
merged_weather_data = new_df.merge(weather_data, how='left', left_index=True, right_index=True)

# Step 3: Interpolate the relevant numerical columns
columns_to_interpolate = ['wind_speed (m/s)', 'mixing_height', 'overall_cloud_cover']
merged_weather_data[columns_to_interpolate] = merged_weather_data[columns_to_interpolate].interpolate(method='time')

# Circular interpolation for wind direction
merged_weather_data['wind_direction_rad'] = np.radians(merged_weather_data['wind_direction (deg)'])
merged_weather_data['wind_dir_sin'] = np.sin(merged_weather_data['wind_direction_rad'])
merged_weather_data['wind_dir_cos'] = np.cos(merged_weather_data['wind_direction_rad'])
merged_weather_data[['wind_dir_sin', 'wind_dir_cos']] = merged_weather_data[['wind_dir_sin', 'wind_dir_cos']].interpolate(method='time')
merged_weather_data['wind_direction_interp_rad'] = np.arctan2(merged_weather_data['wind_dir_sin'], merged_weather_data['wind_dir_cos'])
merged_weather_data['wind_direction_interp_deg'] = np.degrees(merged_weather_data['wind_direction_interp_rad']) % 360
merged_weather_data = merged_weather_data.drop(columns=['wind_direction_rad', 'wind_dir_sin', 'wind_dir_cos', 'wind_direction_interp_rad'])
merged_weather_data['wind_direction (deg)'] = merged_weather_data['wind_direction_interp_deg']
merged_weather_data = merged_weather_data.drop(columns=['wind_direction_interp_deg'])

# Fix the 'Hour' column
solar_data.loc[solar_data['Hour'] == 24, 'Hour'] = 0
solar_data.loc[solar_data['Hour'] == 24, 'Date'] += 1

# Create a datetime column using the 'Date', 'Hour', and 'Minute' columns
solar_data['datetime'] = pd.to_datetime(solar_data['Date'].astype(str) + ' ' + (solar_data['Hour'].astype(int)).astype(str) + ':' + (solar_data['Minute'].astype(int)).astype(str), format='%Y%m%d %H:%M')

# Set the 'datetime' column as the index
solar_data.set_index('datetime', inplace=True)

# Sort the indices of both DataFrames
solar_data = solar_data.sort_index()
merged_weather_data = merged_weather_data.sort_index()

# Find the start and end times of the overlapping period
start_time = max(solar_data.index.min(), merged_weather_data.index.min())
end_time = min(solar_data.index.max(), merged_weather_data.index.max())

# Truncate both DataFrames to keep only the overlapping period
solar_data_overlap = solar_data.truncate(before=start_time, after=end_time)
merged_weather_data_overlap = merged_weather_data.truncate(before=start_time, after=end_time)

# Merge the two DataFrames on the datetime index
weather_final = solar_data_overlap.merge(merged_weather_data_overlap, left_index=True, right_index=True)

# Set a very low wind speed value
very_low_wind_speed = 0.001

# Create an empty DataFrame with the same columns as weather_final
modified_weather_final = pd.DataFrame(columns=weather_final.columns)

# Iterate through the rows of weather_final
for i, row in weather_final.iterrows():
    if row['wind_speed (m/s)'] == 0:
        # Create 15 rows for each minute in the 15-minute interval with very_low_wind_speed and random wind directions
        for j in range(15):
            new_row = row.copy()
            new_row['wind_speed (m/s)'] = very_low_wind_speed
            new_row['wind_direction (deg)'] = random.uniform(0, 360)
            new_index = i + pd.Timedelta(minutes=j)
            modified_weather_final.loc[new_index] = new_row
    else:
        modified_weather_final.loc[i] = row

# Sort the DataFrame by index
modified_weather_final.sort_index(inplace=True)

# Calculate the time intervals between each row and the previous row in the 'datetime' column
modified_weather_final['time_interval'] = modified_weather_final.index.to_series().diff()

# Set the 'time_interval' value for the first row to 15 minutes since there is no previous row to calculate the interval from
modified_weather_final.loc[modified_weather_final.index[0], 'time_interval'] = pd.Timedelta(minutes=15)

class Stability:
    def __init__(self, incidence, cloud_cover, ceiling_height, wind_speed):
        self.beta = incidence
        self._U = wind_speed
        self.CC = cloud_cover
        self.H = ceiling_height

    def calculate_insolation_index(self):
        if self.beta < 15:
            return 1
        elif 15 <= self.beta < 35:
            return 2
        elif 35 <= self.beta < 60:
            return 3
        else:
            return 4

    """Takes angle of solar incidence relative to the horizontal (ground) in degrees
         and outputs Insolation Index. First requirement for Turner's method."""

    def nighttime(self):
        if self.beta > 5:
            return False
        else:
            return True

    def calculate_nri(self):
        insolation_index = self.calculate_insolation_index()
        if self.CC == 100 and self.H < 7000:
            return 0
        else:
            if self.nighttime():
                if self.CC <= 40:
                    return -2
                else:
                    return -1
            else:
                if self.CC <= 50:
                    return insolation_index
                else:
                    if self.H < 7000:
                        insolation_index += (-2)
                    else:
                        if self.H < 16000:
                            insolation_index += (-1)
                            if self.CC == 100:
                                insolation_index += (-1)
                    if insolation_index < 1:
                        return 1
                    else:
                        return insolation_index

    """
        Functions to retrieve the Normalized Response Index
        Uses angle of solar incidence to calculate Insolation Index using parent class.
        Adjusts Insolation Index according to cloud cover expressed as a percent, the ceiling height
        in feet.
        The time of day is compared to the time at which the sun rises and sets to determine whether
        it is nighttime. Sunrise and sunset are currently just placeholders, and the specific times will need
        to be input later when I actually look them up.
        """

    def calculate_stability(self):
        nri = self.calculate_nri()
        speed = self._U
        if 0 <= speed < 0.8:
            if nri == -2:
                return "G"
            elif nri == -1:
                return "F"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "C"
            elif nri == 2:
                return "B"
            elif nri >= 3:
                return "A"
        elif 0.8 <= speed < 1.9:
            if nri == -2:
                return "G"
            elif nri == -1:
                return "F"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "C"
            elif nri == 2:
                return "B"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "A"
        elif 1.9 <= speed < 2.9:
            if nri == -2:
                return "F"
            elif nri == -1:
                return "E"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "A"
        elif 2.9 <= speed < 3.4:
            if nri == -2:
                return "F"
            elif nri == -1:
                return "E"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "B"
        elif 3.4 <= speed < 3.9:
            if nri == -2:
                return "E"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "B"
            elif nri == 4:
                return "B"
        elif 3.9 <= speed < 4.9:
            if nri == -2:
                return "E"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "C"
            elif nri == 4:
                return "B"
        elif 4.9 <= speed < 5.5:
            if nri == -2:
                return "E"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "D"
            elif nri == 3:
                return "C"
            elif nri == 4:
                return "C"
        elif 5.5 <= speed < 6:
            if nri == -2:
                return "D"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "C"
            elif nri == 3:
                return "C"
            elif nri == 4:
                return "C"
        else:
            if nri == -2:
                return "D"
            elif nri == -1:
                return "D"
            elif nri == 0:
                return "D"
            elif nri == 1:
                return "D"
            elif nri == 2:
                return "D"
            elif nri == 3:
                return "D"
            elif nri == 4:
                return "C"


def calculate_stability_class(row):
    stability = Stability(
        incidence=row['Elevation angle'],
        cloud_cover=row['overall_cloud_cover'],
        ceiling_height=row['mixing_height'],
        wind_speed=row['wind_speed (m/s)']
    )
    return stability.calculate_stability()

modified_weather_final['stability_class'] = modified_weather_final.apply(calculate_stability_class, axis=1)


compounds = ["Total Vinyl Chloride", "Total HCl 100%", "Total HCl 52%", "Total HCl 20%", "Total Phosgene 7",
                 "Total Phosgene 0.7", "Total Phosgene 0.07", "ethyl_acryl_100", "ethyl_acryl_80", "ethyl_acryl_50",
                 "ethyl_acryl_20", "butyl_acryl", "PM2.5", "PM10", "Total Dioxin 20%", "Total Dioxin 50%",
             "Total Dioxin 80%", "Total Dioxin 100%"]
PAC1 = [640, 2.7, 2.7, 2.7, .11, .11, .11, 110, 110, 110, 110, 43, .035, .12, .00013, .00013, .00013, .00013]  #mg/m3
PAC2 = [3100, 33, 33, 33, 1.2, 1.2, 1.2, 870, 870, 870, 870, 680, .035, .15, .0014, .0014, .0014, .0014]
PAC3 = [12000, 150, 150, 150, 3, 3, 3, 1100, 1100, 1100, 1100, 2500, .035, .15, .0085, .0085, .0085, .0085]
exposure_limits = pd.DataFrame({'Compounds': compounds, 'PAC1': PAC1, 'PAC2': PAC2, 'PAC3': PAC3})


cleaned_data_folder = "Cleaned_Data"
if not os.path.exists(cleaned_data_folder):
    os.makedirs(cleaned_data_folder)

modified_weather_final.to_csv(os.path.join(cleaned_data_folder, 'cleaned_weather_data.csv'), index=True)
flows_and_rates.to_csv(os.path.join(cleaned_data_folder, 'cleaned_flows_and_rates.csv'), index=True)
exposure_limits.to_csv(os.path.join(cleaned_data_folder, 'exposure_limits.csv'), index=True)
demographic_data.to_csv(os.path.join(cleaned_data_folder, 'cleaned_demographic_data.csv'), index=True)


