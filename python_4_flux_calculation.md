---
title: "1. BASICS OF PYTHON"
layout: default
nav_order: 5
---

# 03. **Fluxes Calculation**
In this tutorial, we're going to analyze the data you collected on your field trip to the Lüner forest! Your instruments measured raw gas concentrations, but as ecologists, we need to turn that into gas fluxes. Why? Because fluxes represent a rate—the speed at which gases are being exchanged. With CO₂ fluxes, we can estimate crucial metrics like ecosystem respiration (RECO) and net ecosystem exchange (NEE). With fluxes of a potent greenhouse gas like Nitrous Oxide (N₂O), we can understand a key part of the nitrogen cycle. This guide will walk you through the entire process: from cleaning the raw concentration data, to calculating meaningful fluxes, and finally to comparing the results between different land cover types.
> **Notice:**
>
>In all following sections I will insert some code snippets. You are very much encouraged to copy and paste them with the button on the top right and run them in your IDE (e.g. Spyder, vscode).
### Table of Contents
1. [Read in and Merge Data Files](#1-Loading-and-Exploring-Raw-Data)
2. [Loading and Exploring Raw Data](#2-Loading-and-Exploring-Raw-Data)
3. [Filtering and Cleaning](#3-Filtering-and-Cleaning)
4. [Understanding the Data Pattern](#4-Understanding-the-Data-Pattern)
5. [Calculating Flux for a Single Plot](#5-Calculating-Flux-for-a-Single-Plot)
6. [Automating Calculations for all plots](#6-Automating-Calculations-for-all-plots)
7. [Comparing Results](#7-Comparing-Results)


## 1.Read in and Merge Data Files
Different from the file reading we did before, the file reading of the raw raw gas concentrations data is a bit more complex. Except our real data, the file also contains metadata of the measurement (timezone, device model, etc), see the following figure, which is useless for this analysis. Therefore, we need to remove it.
![Metadata](/assets/images/05/Metadata.png)

    with open("./BAI_StudyProject_LuentenerWald/raw_data/TG20-01072-2025-08-15T110000.data.txt") as f:
      file_content = f.read()

We will use Python with the pandas, matplotlib, seaborn, and scipy libraries.
## 2.Loading and Exploring Raw Data
Step 1: Loading and Initial Exploration of Raw Data
First, we need to load our data from its raw text file format into a pandas DataFrame. This file has a custom header, so we need to parse it carefully. The function below handles reading the file, skipping the metadata lines, and converting the date and time columns into a proper timestamp.
    
    import pandas as pd
    import numpy as np
    import io
    import matplotlib.pyplot as plt
   
    def load_and_clean_data(filepath):
        """
        Loads raw data from a text file, cleans it, and returns a DataFrame.
        """
        
        with open(filepath) as f:
            file_content = f.read()

        lines = file_content.strip().split('\n')
        header_index = next(i for i, line in enumerate(lines) if line.startswith('DATAH'))
        data_start_index = header_index + 2
        headers = lines[header_index].split('\t')

        df_raw = pd.read_csv(
            io.StringIO('\n'.join(lines[data_start_index:])),
            sep='\t',
            header=None,
            names=headers,
            na_values='nan'
        )

        if 'DATAH' in df_raw.columns:
            df_raw = df_raw.drop(columns=['DATAH'])

        if 'DATE' in df_raw.columns and 'TIME' in df_raw.columns:
            df_raw['Timestamp'] = pd.to_datetime(df_raw['DATE'] + ' ' + df_raw['TIME'])
            df_raw = df_raw.drop(columns=['DATE', 'TIME'])
            df_raw = df_raw.set_index('Timestamp')
        
        print("Raw data loaded and cleaned successfully.")
        return df_raw

# --- Load the data ---
filepath = "./BAI_StudyProject_LuentenerWald/raw_data/TG20-01072-2025-08-15T110000.data.txt"
df_raw = load_and_clean_data(filepath)

print("\nFirst 5 rows of raw data:")
print(df_raw.head())

# --- Visualize the raw data ---
fig, ax = plt.subplots(layout='constrained', figsize=(20, 5))
ax.plot(df_raw.index, df_raw['N2O'], label='N2O Concentration (ppb)')
ax.set_xlabel('Time')
ax.set_ylabel('N2O Concentration (ppb)')
ax.set_title('Raw N2O Concentration Over Time')
plt.show()
<!-- Placeholder for image -->
As you can see from the plot, the raw data is very noisy. There are several negative values and some extremely large spikes. These are physically impossible and are likely due to sensor errors or electrical interference. We cannot calculate meaningful fluxes from this data without cleaning it first.
Step 2: Data Filtering and Cleaning
To remove these outliers, we'll use a simple but effective quantile filter. We'll calculate the 10th and 90th percentiles of the N₂O concentration and discard any data points that fall outside this range. This will effectively chop off the extreme high and low noise.
code
Python
# Calculate the 10th and 90th percentiles
p_10 = df_raw.N2O.quantile(0.10)
p_90 = df_raw.N2O.quantile(0.90)

# Apply the filter
df_filtered = df_raw[(df_raw.N2O >= p_10) & (df_raw.N2O <= p_90)]

# Visualize the filtered data
fig, ax1 = plt.subplots(layout='constrained', figsize=(20, 5))
ax1.scatter(df_filtered.index, df_filtered['N2O'], label='N2O Concentration (ppb)', s=5) # s=5 makes points smaller
ax1.set_xlabel('Time')
ax1.set_ylabel('N2O Concentration (ppb)')
ax1.set_title('Filtered N2O Concentration Over Time')
plt.show()```

 <!-- Placeholder for image -->

This looks much better! The erratic noise is gone, and a clear pattern has emerged.

## Step 3: Understanding the Data Pattern

The filtered data shows a repeating pattern which is the signature of the **static chamber method**:

*   **Baseline (Ambient Air)**: The long, relatively flat periods show the baseline N₂O concentration in the ambient air.
*   **Concentration Increase (Chamber Closed)**: The sections where the concentration rises steadily and linearly are the actual measurements. This occurs when the chamber is placed over the soil, trapping the gases being emitted. The rate of this increase is what we need to calculate the flux.
*   **Sudden Drop (Chamber Opened)**: The sharp vertical drops occur when a measurement is finished, and the chamber is lifted from the ground, exposing the sensor to ambient air again.
*   **Leveling Off**: If a chamber is left on the ground for too long, the gas concentration inside can build up, altering the pressure gradient between the soil and the chamber air. This can cause the rate of increase to slow down and "level off." For this reason, it's crucial to use the initial, linear part of the increase for our flux calculation.

## Step 4: Calculating Flux for a Single Measurement

To calculate a flux, we need to combine the gas concentration data with **metadata** about our measurement setup. This includes the start and end times of each chamber placement and the physical dimensions of the chamber (like collar height).

First, let's define our flux calculation function, which is based on the Ideal Gas Law.

```python
from scipy import stats
import seaborn as sns

# Define key physical and experimental constants
R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)

def calculate_flux(slope_ppb_s, temp_k, pressure_pa, V_over_A):
    """
    Calculate the N2O flux in µmol m⁻² s⁻¹.
    """
    # Convert slope from ppb/s to ppm/s
    ppm_per_second = slope_ppb_s / 1000

    # Calculate molar density (mol/m³) using the ideal gas law (n/V = P/RT)
    molar_density = pressure_pa / (R * temp_k)

    # Flux (µmol m⁻² s⁻¹) = dC/dt [ppm/s] * V/A [m] * molar_density [mol/m³] * 1e6 [µmol/mol]
    flux_umol_m2_s = ppm_per_second * V_over_A * molar_density * 1e6

    return flux_umol_m2_s
Now, let's create a metadata DataFrame. In a real project, you would load this from a field notebook (e.g., a CSV file). This file tells us when and where each measurement took place. We also calculate the crucial Volume-to-Area (V_over_A) ratio for the chamber at each plot.
code
Python
# --- Create Metadata ---
measurement_info = {
    'plot_id': [1, 2],
    'land_use': ['forest', 'forest'],
    'start_time': ['2025-08-15 12:04:00', '2025-08-15 12:13:00'],
    'end_time': ['2025-08-15 12:09:30', '2025-08-15 12:18:30'],
    'collar_height_m': [0.055, 0.061],
    'chamber_height_m': [0.40, 0.40],
    'chamber_radius_m': [0.2, 0.2]
}
metadata_df = pd.DataFrame(measurement_info)

# Calculate the volume-to-area ratio (V/A)
metadata_df['chamber_area_m2'] = np.pi * metadata_df['chamber_radius_m']**2
metadata_df['total_height_m'] = metadata_df['collar_height_m'] + metadata_df['chamber_height_m']
metadata_df['chamber_volume_m3'] = metadata_df['total_height_m'] * metadata_df['chamber_area_m2']
metadata_df['V_over_A'] = metadata_df['chamber_volume_m3'] / metadata_df['chamber_area_m2']

print("\nMetadata created successfully:")
print(metadata_df)
Now, let's isolate the data for the first plot and perform a linear regression to find the slope (the rate of N₂O increase).
code
Python
# --- Select data for the first plot ---
plot_info = metadata_df.iloc[0]
start_time = pd.to_datetime(plot_info['start_time'])
end_time = pd.to_datetime(plot_info['end_time'])

measurement_data = df_filtered[(df_filtered.index >= start_time) & (df_filtered.index < end_time)]

# Create an 'elapsed_seconds' column for the regression
measurement_data['elapsed_seconds'] = (measurement_data.index - start_time).total_seconds()

# --- Perform linear regression ---
slope, intercept, r_value, p_value, std_err = stats.linregress(
    x=measurement_data['elapsed_seconds'],
    y=measurement_data['N2O']
)

r_squared = r_value**2

# --- Plot the regression line ---
fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
ax.scatter(measurement_data['elapsed_seconds'], measurement_data['N2O'], label='N2O Data')
ax.plot(measurement_data['elapsed_seconds'], intercept + slope * measurement_data['elapsed_seconds'], 'r', label='Fitted line')
ax.set_xlabel('Elapsed Time (s)')
ax.set_ylabel('N2O Concentration (ppb)')
ax.set_title(f'Linear Regression for Plot {plot_info["plot_id"]} (R²={r_squared:.2f})')
plt.legend()
plt.show()

# --- Calculate the flux ---
# We'll assume a constant temperature and pressure for this example.
temp_k = 293.15      # 20°C in Kelvin
pressure_pa = 101325 # Standard atmospheric pressure in Pascals
V_over_A = plot_info['V_over_A']

flux = calculate_flux(slope, temp_k, pressure_pa, V_over_A)

print(f"\nResults for Plot {plot_info['plot_id']}:")
print(f"  - Slope (ppb/s): {slope:.4f}")
print(f"  - R-squared: {r_squared:.4f}")
print(f"  - Calculated Flux (µmol m⁻² s⁻¹): {flux:.4f}")
The plot shows our data and the line of best fit. The R-squared value tells us how well the line fits the data (a value > 0.9 is good). The slope is the crucial dC/dt (change in concentration over time) that we feed into our calculate_flux function.
Step 5: Automating Flux Calculations for All Plots
Now we can wrap this logic in a loop to process all the measurements defined in our metadata_df. We will also include a Quality Control (QC) step: we only accept measurements with a good linear fit (R² > 0.9), a statistically significant slope (p-value < 0.05), and a positive slope (since we expect emissions, not uptake).
code
Python
# --- Flux Calculation Loop ---
results = []

for index, row in metadata_df.iterrows():
    start_time = pd.to_datetime(row['start_time'])
    end_time = pd.to_datetime(row['end_time'])

    # Select the data for this specific time window
    measurement_data = df_filtered[(df_filtered.index >= start_time) & (df_filtered.index < end_time)].copy()
    
    if len(measurement_data) < 10:
        print(f"Skipping plot {row['plot_id']} due to insufficient data.")
        continue

    measurement_data['elapsed_seconds'] = (measurement_data.index - start_time).total_seconds()

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x=measurement_data['elapsed_seconds'],
        y=measurement_data['N2O']
    )

    # --- Quality Control (QC) ---
    r_squared = r_value**2
    if r_squared < 0.90 or p_value > 0.05 or slope < 0:
        flux_umol_m2_s = 0  # Set flux to 0 if QC fails
        qc_pass = False
    else:
        qc_pass = True
        temp_k = 293.15
        pressure_pa = 101325
        V_over_A = row['V_over_A']
        flux_umol_m2_s = calculate_flux(slope, temp_k, pressure_pa, V_over_A)

    # Store the results
    results.append({
        'plot_id': row['plot_id'],
        'land_use': row['land_use'],
        'slope_ppb_s': slope,
        'r_squared': r_squared,
        'p_value': p_value,
        'qc_pass': qc_pass,
        'N2O_flux_umol_m2_s': flux_umol_m2_s
    })

# Convert the results list to a final DataFrame
flux_results_df = pd.DataFrame(results)

print("\nFlux calculation complete:")
print(flux_results_df)
This final DataFrame contains all the information we need: the calculated flux for each plot, the quality of the regression, and whether it passed our QC checks.
Step 6: Comparing Fluxes for Different Land Covers
The final step is to visualize our results. We want to see if there is a difference in N₂O emissions between our different land uses. A boxplot is a perfect way to do this.
Let's expand our metadata to include a hypothetical "grassland" land use to make the comparison more interesting.
code
Python
# --- Re-create metadata with two land uses for comparison ---
measurement_info_full = {
    'plot_id': [1, 2, 3, 4],
    'land_use': ['forest', 'forest', 'grassland', 'grassland'],
    'start_time': ['2025-08-15 12:04:00', '2025-08-15 12:13:00', '2025-08-15 12:18:00', '2025-08-15 12:23:00' ],
    'end_time': ['2025-08-15 12:09:30', '2025-08-15 12:18:30', '2025-08-15 12:21:00', '2025-08-15 12:26:00'],
    'collar_height_m': [0.055, 0.061, 0.045, 0.049],
    'chamber_height_m': [0.40] * 4,
    'chamber_radius_m': [0.2] * 4
}
# NOTE: You would need to re-run the loop in Step 5 with this new metadata
# to generate the full flux_results_df for this plot.
# For this tutorial, we will create a sample final DataFrame.

# Sample final DataFrame (replace with your actual results)
final_data = {
    'land_use': ['forest', 'forest', 'grassland', 'grassland'],
    'N2O_flux_umol_m2_s': [0.012, 0.015, 0.025, 0.028] # Example values
}
flux_results_df_sample = pd.DataFrame(final_data)


# --- Visualization ---
plt.figure(figsize=(10, 7))
sns.boxplot(data=flux_results_df_sample, x='land_use', y='N2O_flux_umol_m2_s', palette='viridis')
sns.stripplot(data=flux_results_df_sample, x='land_use', y='N2O_flux_umol_m2_s', color='black', size=8, jitter=True, alpha=0.7)

plt.title('N₂O Flux by Land Use Type', fontsize=16)
plt.xlabel('Land Use', fontsize=12)
plt.ylabel('N₂O Flux (µmol m⁻² s⁻¹)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

``` <!-- Placeholder for image -->

This final visualization allows us to easily compare the distribution of fluxes from the forest and grassland plots. From this example plot, we could conclude that the grassland plots tend to have higher N₂O emissions than the forest plots.
