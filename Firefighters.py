import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def dC_dt(t, C, Q, k, adv_rate):
    dC_dt = (Q / V_box) - (k * C) - (adv_rate * C)
    return dC_dt

def box_model(flows_and_rates, weather_data, box_dimensions, dt, simulation_duration):
    # Resample the weather data to a uniform time step
    weather_data_resampled = weather_data.resample(f'{dt}T').interpolate()

    # Calculate the volume of the box
    V_box = box_dimensions['length'] * box_dimensions['width'] * box_dimensions['height']

    # Initialize the concentrations dataframe
    num_steps = int(simulation_duration / dt)
    time_index = pd.date_range(start=weather_data_resampled.index[0], periods=num_steps, freq=f'{dt}T')
    concentrations = pd.DataFrame(index=time_index, columns=flows_and_rates.index, data=np.zeros((num_steps, len(flows_and_rates))))

    # Time-stepping loop
    for t_step in range(1, num_steps):
        # Update the wind speed
        wind_speed = weather_data_resampled['wind_speed'].iloc[t_step]

        # Calculate the advection rate
        adv_rate = wind_speed / box_dimensions['length']

        # Solve the ODE system for each compound
        for compound in flows_and_rates.index:
            Q_i = flows_and_rates.loc[compound, 'Q (kg/s)']
            k_i = flows_and_rates.loc[compound, 'k (per sec)']
            C_initial = concentrations.loc[time_index[t_step-1], compound]

            t_span = (time_index[t_step-1], time_index[t_step])
            sol = solve_ivp(dC_dt, t_span, [C_initial], args=(Q_i, k_i, adv_rate), method='RK45', dense_output=True)
            concentrations.loc[time_index[t_step], compound] = sol.sol(time_index[t_step])[0]

    return concentrations

weather_final = pd.read_csv('Cleaned_Data/cleaned_weather_data.csv', index_col=0, parse_dates=True)
start_time = pd.to_datetime('2023-02-03 21:00:00')
end_time = pd.to_datetime('2023-02-04 04:00:00')

weather_truncated = weather_final.truncate(before=start_time, after=end_time)

flows_and_rates = pd.read_csv('Cleaned_Data/cleaned_flows_and_rates.csv', index_col=0)
rows_to_drop = [
    'vin_chlor_4',
    'vin_chlor_1',
    'hcl_4_100',
    'hcl_4_52',
    'hcl_4_20',
    'hcl_1_100',
    'hcl_1_52',
    'hcl_1_20',
    'phos_1_7',
    'phos_1_0.7',
    'phos_1_0.07',
    'phos_4_7',
    'phos_4_0.7',
    'phos_4_0.07'
]

flows_and_rates_filtered = flows_and_rates[~flows_and_rates['plume_name'].isin(rows_to_drop)]
box_dimensions = {'length': 100, 'width': 100, 'height': 100}
simulation_duration_minutes = (weather_truncated.index[-1] - weather_truncated.index[0]).total_seconds() / 60
dt = 1
concentrations = box_model(flows_and_rates_filtered, weather_truncated, box_dimensions, dt, simulation_duration_minutes)
concentrations.to_csv('FireFighterResults/concentrations.csv')
