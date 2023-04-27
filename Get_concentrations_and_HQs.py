import numpy as np
import pandas as pd
import math
import time
from tqdm import tqdm


def create_plume_locations(fcar, car, ethylcar, butylcar,
                           fcar_times, car_times, ethylcar_times, butylcar_times, stack_heights):
    fcarx, fcary = fcar
    carx, cary = car
    ethylcarx, ethylcary = ethylcar
    butylcarx, butylcary = butylcar

    fcar_start, fcar_end = fcar_times
    car_start, car_end = car_times
    ethylcar_start, ethylcar_end = ethylcar_times
    butylcar_start, butylcar_end = butylcar_times
    uncontrolled_h, controlled_h = stack_heights

    plume_data = {
        'plume_name': [],
        'X': [],
        'Y': [],
        'Start_time': [],
        'End_time': [],
        'stack_height': []
    }

    for i in range(29):
        plume_data['plume_name'].append(f"plume_{i+1}")

    plume_x_coords = [fcarx, carx, ethylcarx, ethylcarx, ethylcarx, ethylcarx, butylcarx, fcarx, fcarx, fcarx, fcarx,
                      fcarx, fcarx, fcarx, fcarx, fcarx, fcarx, fcarx, carx,
                      carx, carx, fcarx, fcarx, fcarx, carx, carx, carx, butylcarx, butylcarx]
    plume_y_coords = [fcary, cary, ethylcary, ethylcary, ethylcary, ethylcary, butylcary, fcary, fcary, fcary, fcary,
                      fcary, fcary, fcary, fcary, fcary, fcary, fcary, cary,
                      cary, cary, fcary, fcary, fcary, cary, cary, cary, butylcary, butylcary]
    plume_start_times = [fcar_start, car_start, ethylcar_start, ethylcar_start, ethylcar_start, ethylcar_start,
                         butylcar_start, fcar_start, fcar_start, fcar_start, fcar_start,
                         fcar_start, fcar_start, fcar_start, fcar_start, fcar_start, fcar_start, fcar_start, car_start, car_start, car_start,
                         fcar_start, fcar_start, fcar_start, car_start, car_start, car_start,
                         butylcar_start, butylcar_start]
    plume_end_times = [fcar_end, car_end, ethylcar_end, ethylcar_end, ethylcar_end, ethylcar_end, butylcar_end,
                       fcar_end, fcar_end, fcar_end, fcar_end, fcar_end, fcar_end, fcar_end, fcar_end,
                       fcar_end, fcar_end, fcar_end, car_end, car_end, car_end, fcar_end, fcar_end, fcar_end, car_end,
                       car_end, car_end, butylcar_end, butylcar_end]
    stack_heights = [controlled_h, controlled_h, uncontrolled_h, uncontrolled_h, uncontrolled_h, uncontrolled_h,
                     uncontrolled_h, controlled_h, controlled_h, controlled_h, controlled_h, controlled_h,
                     controlled_h, controlled_h, controlled_h, controlled_h, controlled_h, controlled_h, controlled_h,
                     controlled_h, controlled_h, controlled_h, controlled_h, controlled_h, controlled_h, controlled_h,
                     controlled_h, controlled_h, controlled_h]

    plume_data['X'] = plume_x_coords
    plume_data['Y'] = plume_y_coords
    plume_data['Start_time'] = plume_start_times
    plume_data['End_time'] = plume_end_times
    plume_data['stack_height'] = stack_heights

    plume_locations_df = pd.DataFrame(plume_data)

    return plume_locations_df
class PlumeCoordinateTransform:
    def __init__(self, wind_direction, plume_origin, map_origin):
        self._wind_dir = wind_direction
        self._plume_origin = plume_origin
        self._map_origin = map_origin
        self._dx = self._plume_origin[0] - self._map_origin[0]
        self._dy = self._plume_origin[1] - self._map_origin[1]
        self._theta = np.deg2rad(self._wind_dir)

    def map_to_plume(self, map_coords):
        x_map, y_map = map_coords
        x_plume = self._dx + x_map * math.cos(self._theta) - y_map * math.sin(self._theta)
        y_plume = self._dy + x_map * math.sin(self._theta) + y_map * math.cos(self._theta)
        return (x_plume, y_plume)

    def plume_to_map(self, plume_coords):
        x_plume, y_plume = plume_coords
        x_map = self._map_origin[0] + (x_plume - self._dx) * math.cos(self._theta) + \
                (y_plume - self._dy) * math.sin(self._theta)
        y_map = self._map_origin[1] + (y_plume - self._dy) * math.cos(self._theta) - \
                (x_plume - self._dx) * math.sin(self._theta)
        return (x_map, y_map)


def Sigma_y(sc, d):
    if sc == 'A':
        result = .22 * d * pow((1 + 0.0001 * d), -0.5)
    elif sc == 'B':
        result = .16 * d * pow((1 + 0.0001 * d), -0.5)
    elif sc == 'C':
        result = .11 * d * pow((1 + 0.0001 * d), -0.5)
    elif sc == 'D':
        result = .08 * d * pow((1 + 0.0001 * d), -0.5)
    elif sc == 'E':
        result = .06 * d * pow((1 + 0.0001 * d), -0.5)
    else:
        result = .04 * d * pow((1 + 0.0001 * d), -0.5)

    return result


def Sigma_z(sc, d):
    if sc == 'A':
        result = 0.2 * d
    elif sc == 'B':
        result = 0.12 * d
    elif sc == 'C':
        result = .08 * d * pow((1 + 0.0002 * d), -0.5)
    elif sc == 'D':
        result = .06 * d * pow((1 + 0.0015 * d), -0.5)
    elif sc == 'E':
        result = .03 * d * pow((1 + 0.0003 * d), -1)
    else:
        result = .016 * d * pow((1 + 0.0003 * d), -1)

    return result


def VinylChloride(Q, U, h, x, y, z, decay, sigmaY, sigmaZ):
    if x <= 0:
        return 0
    if sigmaY <= 0:
        sigmaY = 1e-6
    if sigmaZ <= 0:
        sigmaZ = 1e-6
    if U == 0:
        return 0
    else:
        return (Q / U) * math.exp(-decay*x/U) * pow(2 * math.pi * sigmaY * sigmaZ, -1) * \
               math.exp(-0.5 * pow(y / sigmaY, 2)) * \
               (math.exp(-0.5 * pow((z - h) / sigmaZ, 2)) + math.exp(-0.5 * pow((z + h) / sigmaZ, 2)))

def plumewithsettling(Q, U, h, x, y, z, sigmaY, sigmaZ, vg):
    def tilted_center_line(U, x, vg):
        return h + vg * x / U
    if x <= 0:
        return 0
    if sigmaY <= 0:
        sigmaY = 1e-6
    if sigmaZ <= 0:
        sigmaZ = 1e-6
    if U == 0:
        return 0
    else:
        concentration = (Q / U) * pow(2*math.pi*sigmaY*sigmaZ, -1)* math.exp(-0.5*pow(y/sigmaY, 2)) * math.exp(-0.5*pow(z - tilted_center_line(U, x, vg) / sigmaZ, 2))
        return concentration
def get_concentrations_dict(demographic_data, weather_final, flows_and_rates, plume_locations):
    stopwatch = time.time()
    plume_locations_df = pd.DataFrame(plume_locations)
    plume_locations_df.reset_index(inplace=True)
    plume_locations_df.rename(columns={'index': 'plume_name'}, inplace=True)

    plume_locations_df.drop('plume_name', axis=1, inplace=True)

    flows_and_rates_merged = flows_and_rates.merge(plume_locations_df, left_index=True, right_index=True, how='left')

    concentrations_dict = {}
    total_rows = len(demographic_data)

    for index, row in tqdm(demographic_data.iterrows(), total=total_rows, desc="Progress"):
        name = row['NAME']
        x = row['x']
        y = row['y']
        concentration_data = weather_final.copy()

        for index_c, row_c in concentration_data.iterrows():
            wind_speed = row_c['wind_speed (m/s)']
            wind_dir = row_c['wind_direction (deg)']
            stability = row_c['stability_class']

            for index_p, row_p in flows_and_rates_merged.iterrows():
                plume_origin = (row_p["X"], row_p['Y'])
                start_time = row_p['Start_time']
                end_time = row_p['End_time']
                h = row_p['stack_height']
                coord_transform = PlumeCoordinateTransform(wind_dir, plume_origin, (0, 0))
                downwind_distance = coord_transform.map_to_plume((x, y))[0]
                crosswind_distance = coord_transform.map_to_plume((x, y))[1]
                sigmaY = Sigma_y(stability, downwind_distance)
                sigmaZ = Sigma_z(stability, downwind_distance)
                release_rate = row_p["Q (kg/s)"]
                decay_rate = row_p["k (per sec)"]
                vg = row_p["settling"]
                if vg == 0:
                    if index_c >= start_time and index_c <= end_time:
                        concentration = VinylChloride(release_rate, wind_speed, h, downwind_distance,
                                                      crosswind_distance, 0, decay_rate, sigmaY, sigmaZ)
                    else:
                        concentration = 0

                else:
                    if index_c >= start_time and index_c <= end_time:
                        concentration = plumewithsettling(release_rate, wind_speed, h, downwind_distance,
                                                          crosswind_distance, 0, sigmaY, sigmaZ, vg)
                    else: concentration = 0

                source_name = row_p["plume name"]
                concentration_data.loc[index_c, source_name] = concentration

        concentrations_dict[name] = concentration_data

    for name, concentration_data in concentrations_dict.items():
        concentration_data['Total Vinyl Chloride'] = concentration_data['vin_chlor_4'] + concentration_data['vin_chlor_1']

        for x in [100, 52, 20]:
            concentration_data[f'Total HCl {x}%'] = concentration_data[f'hcl_4_{x}'] + concentration_data[f'hcl_1_{x}']

        for x in [7, 0.7, 0.07]:
            x_str = f'{x:.2f}'.rstrip('0').rstrip('.')
            concentration_data[f'Total Phosgene {x_str}'] = concentration_data[f'phos_4_{x_str}'] + concentration_data[f'phos_1_{x_str}']
        for x in [100, 80, 50, 20]:
            concentration_data[f'Total Dioxin {x}%'] = concentration_data[f'dioxin_4_{x}'] + concentration_data[f'dioxin_1_{x}']

    stopwatch2 = time.time()
    elapsed_time = stopwatch2 - stopwatch
    print(f"Execution time: {elapsed_time:.2f} seconds")

    return concentrations_dict



def add_time_weighted_averages(concentrations_dict):
    time_intervals = [10, 15, 30, 60, 8 * 60, 24 * 60]  # Time intervals in minutes

    for name, concentration_data in concentrations_dict.items():
        # Calculate the time differences between rows in minutes
        time_diffs = concentration_data.index.to_series().diff().fillna(pd.Timedelta(0)) / np.timedelta64(1, 'm')

        for interval in time_intervals:
            for compound in ["Total Vinyl Chloride", "Total HCl 100%", "Total HCl 52%", "Total HCl 20%",
                             "Total Phosgene 7", "Total Phosgene 0.7",
                             "Total Phosgene 0.07", "ethyl_acryl_100",
                             "ethyl_acryl_80", "ethyl_acryl_50", "ethyl_acryl_20", "butyl_acryl", "PM2.5",
                             "PM10", "Total Dioxin 20%", "Total Dioxin 50%", "Total Dioxin 80%", "Total Dioxin 100%"]:
                # Create a new DataFrame to store the rolling sum and time
                rolling_data = pd.DataFrame(index=concentration_data.index)
                rolling_data['rolling_sum'] = (concentration_data[compound] * time_diffs).cumsum()
                rolling_data['rolling_time'] = time_diffs.cumsum()

                # Subtract the values at (t - interval) from the rolling sum and time
                rolling_data['rolling_sum'] -= rolling_data['rolling_sum'].shift(periods=1, fill_value=0).where(rolling_data['rolling_time'].shift(periods=1, fill_value=0) >= (rolling_data['rolling_time'] - interval))
                rolling_data['rolling_time'] -= rolling_data['rolling_time'].shift(periods=1, fill_value=0).where(rolling_data['rolling_time'].shift(periods=1, fill_value=0) >= (rolling_data['rolling_time'] - interval))

                # Calculate the time-weighted average
                concentration_data[f"{compound} TWA {interval} min"] = rolling_data['rolling_sum'] / rolling_data['rolling_time']

    return concentrations_dict


def create_twa_dataframes(concentrations_dict, exposure_limits):
    compounds = ["Total Vinyl Chloride", "Total HCl 100%", "Total HCl 52%", "Total HCl 20%", "Total Phosgene 7",
                 "Total Phosgene 0.7", "Total Phosgene 0.07", "ethyl_acryl_100", "ethyl_acryl_80", "ethyl_acryl_50",
                 "ethyl_acryl_20", "butyl_acryl", "PM2.5", "PM10", "Total Dioxin 20%", "Total Dioxin 50%",
                 "Total Dioxin 80%", "Total Dioxin 100%"]

    twa_dataframes = {}

    start_time = time.time()

    for compound in tqdm(compounds, desc="Processing compounds"):
        if compound in ["PM2.5", "PM10"]:
            twa_column = f"{compound} TWA 1440 min"
        else:
            twa_column = f"{compound} TWA 60 min"

        # Get the corresponding PAC values from exposure_limits
        pac1_value = exposure_limits.loc[exposure_limits['Compounds'] == compound, 'PAC1'].values[0]
        pac2_value = exposure_limits.loc[exposure_limits['Compounds'] == compound, 'PAC2'].values[0]
        pac3_value = exposure_limits.loc[exposure_limits['Compounds'] == compound, 'PAC3'].values[0]

        dataframes_per_time = {}

        for time_interval in concentrations_dict[next(iter(concentrations_dict))].index:
            columns = []

            for name, concentration_data in concentrations_dict.items():
                twa_value = concentration_data.loc[time_interval, twa_column]
                hazard_index_pac1 = (twa_value*1000) / pac1_value
                hazard_index_pac2 = (twa_value*1000) / pac2_value
                hazard_index_pac3 = (twa_value*1000) / pac3_value

                series_data = pd.Series([twa_value, hazard_index_pac1, hazard_index_pac2, hazard_index_pac3],
                                        index=[twa_column, f"{compound} Hazard Index PAC1",
                                               f"{compound} Hazard Index PAC2",
                                               f"{compound} Hazard Index PAC3"])
                columns.append(pd.Series(series_data, name=name))

            new_df = pd.concat(columns, axis=1).T
            dataframes_per_time[time_interval] = new_df

        twa_dataframes[compound] = dataframes_per_time

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

    return twa_dataframes