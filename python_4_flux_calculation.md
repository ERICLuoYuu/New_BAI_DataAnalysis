---
title: "3. FLUX CALCULATION"
layout: default
nav_order: 4
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
Different from the simple CSV files we might have worked with before, the raw data from the gas analyzer is more complex. When you open the file, you'll see it contains two parts:
A metadata header: This block at the top contains useful information about the measurement (like timezone, device model, etc.), but we don't need it for our flux calculations.
The data block: This is the core data we need, with columns for date, time, and gas concentrations.
Our first challenge is to programmatically read only the data block and ignore the metadata.

![Metadata](/assets/images/python/5/Metadata.png)

To do this, we'll need the pandas library for creating our DataFrame and the io library, we need to import them.
```python
import pandas as pd
import io
```
    
Our strategy will be to read the file line-by-line, find the start of the data, and then pass only those lines to pandas.
### 1.1 Reading and Parsing the File
First, we read the entire file into a single string, and then split that string into a list of individual lines. This gives us the flexibility to find our data "landmarks."

```python

# Read in raw data as a string
with open("./BAI_StudyProject_LuentenerWald/raw_data/TG20-01072-2025-08-15T110000.data.txt") as f:
    file_content = f.read()

# Split the string into a list of lines. 
# '\n' is the special character for a newline.
lines = file_content.strip().split('\n')
```

Next, we need to find the exact line that contains our column headers. Looking at the file, we know this line always starts with the word DATAH. We can write a short command to find the index of that line.

```python
# This code searches through our list 'lines' and gets the index of the first line that starts with 'DATAH'
header_index = next(i for i, line in enumerate(lines) if line.startswith('DATAH'))

# The actual data starts 2 lines after the header line (to skip the "DATAU" units line)
data_start_index = header_index + 2

# Now we can grab the headers themselves from that line. The values are separated by tabs ('\t').
headers = lines[header_index].split('\t')
```
### 1.2 Using io.StringIO to Read Our Cleaned Data
The pd.read_csv() function is built to read from a file. We don't have a clean file; we have a list of Python strings (lines) that we've already processed.
So, how do we make pandas read from our list? We use io.StringIO to trick pandas. It takes our cleaned-up data lines and presents them to pandas as if they were a file stored in the computer's memory.
>Info:
>The Python io module helps us manage data streams. io.StringIO specifically allows us to treat a regular text string as a >file. This is incredibly useful when you need to pass text data to a function that expects a file, just like we're doing >with pd.read_csv().

```python
# Join our data lines back into a single string, separated by newlines
data_string = '\n'.join(lines[data_start_index:])
# Read the data string into a DataFrame
df_raw = pd.read_csv(
    io.StringIO(data_string),  # Treat our string as a file
    sep='\t',                  # Tell pandas the data is separated by tabs
    header=None,               # We are providing the headers ourselves, so there isn't one in the data
    names=headers,             # Use the 'headers' list we extracted earlier
    na_values='nan'            # Recognize 'nan' strings as missing values
)
```

### 1.3 Data Formatting
The last step is to tidy up the DataFrame. We will:
Remove the useless DATAH column.
Combine the separate DATE and TIME columns into a single Timestamp object. This is crucial for time-series analysis.
Set this new Timestamp as the DataFrame's index, which makes plotting and selecting data by time much easier.

```python
# Drop the first column which is just the 'DATAH' label
if 'DATAH' in df_raw.columns:
    df_raw = df_raw.drop(columns=['DATAH'])

# Combine 'DATE' and 'TIME' into a proper Timestamp and set it as the index
if 'DATE' in df_raw.columns and 'TIME' in df_raw.columns:
    df_raw['Timestamp'] = pd.to_datetime(df_raw['DATE'] + ' ' + df_raw['TIME'])
    df_raw = df_raw.drop(columns=['DATE', 'TIME'])
    df_raw = df_raw.set_index('Timestamp')

print("Data loaded and formatted successfully!")
df_raw.head()
```
Great! Now, we have successfully read in and formatted our raw data.
However, think about our field campaigns. We went out several times and generate a new data file for each trip. If we wanted to analyze all of them, we would have to copy and paste our loading code multiple times.
To avoid repetition and make our code cleaner and more reliable, it's a best practice to wrap a reusable process into a function. Let's turn our loading and cleaning steps into a function called load_raw_data.

>Your Task: Try to write this function yourself!
>
>Tip: The function will need to accept one argument: the filepath of the file you want to open.

<details><summary>Solution!</summary>
Note: how it's the exact same logic as before, just defined within a def block.
    
```python
   def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw data from a text file, remove metadata, and returns a DataFrame.

    Parameters:
    - filepath (str): The path to the input data file.

    Returns:
    - pd.DataFrame: A cleaned DataFrame with a DatetimeIndex.
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
```
</details>

Now that we have our powerful load_raw_data function, we can easily handle data from multiple field trips. Instead of copying code, we can simply call our function in a loop.
First, we create a list of all the file paths we want to load. Then, we can loop through this list, call our function for each path, and store the resulting DataFrames in a new list.

```python
# First, let's list all the files we want to load.
# Make sure the file paths are complete and correct.
base_path = "./BAI_StudyProject_LuentenerWald/raw_data/"
file_names = [
    'TG20-01072-2025-08-15T110000.data.txt', 
    'TG20-01072-2025-08-16T110000.data.txt' # A hypothetical second file
]

# Create the full file paths
full_file_paths = [base_path + name for name in file_names]

# Create an empty list to hold the loaded DataFrames
raw_data_list = []

# Loop through each path, load the data, and append it to our list
for path in full_file_paths:
    df = load_raw_data(path)
    raw_data_list.append(df)

print(f"\nSuccessfully loaded {len(raw_data_list)} data files.")
```

The loop above is clear and correct. However, a more concise way to write this in Python is with a list comprehension. It achieves the exact same result in a single, readable line:

```python
raw_data_list = [load_raw_data(path) for path in full_file_paths]
```

For our flux calculations to be accurate, we need more than just gas concentrations. The Ideal Gas Law, which is the basis of the calculation, requires the ambient air temperature and air pressure at the time of each measurement.
We will use the same workflow as before: load each file and then combine them.

>Your Task:
>You have two Excel files containing air temperature and two files for air pressure.
>Create lists of the file paths for the temperature and pressure data.
>Load each Excel file into a pandas DataFrame. Try using a list comprehension as we learned before!

<details>
<summary>Click here for the solution!</summary>
    
```Python
# We assume the base path is the same as before
base_path = "./BAI_StudyProject_LuentenerWald/raw_data/"

# --- Load Air Temperature Data ---
file_names_Ta = [
    'air_temperature_2025-08-15.xlsx', 
    'air_temperature_2025-08-16.xlsx' 
]
full_file_paths_Ta = [base_path + name for name in file_names_Ta]
ta_data_list = [pd.read_excel(path) for path in full_file_paths_Ta]
print(f"Successfully loaded {len(ta_data_list)} air temperature files.")

# --- Load Air Pressure Data ---
file_names_Pa = [
    'air_pressure_2025-08-15.xlsx', 
    'air_pressure_2025-08-16.xlsx' 
]
full_file_paths_Pa = [base_path + name for name in file_names_Pa]
pa_data_list = [pd.read_excel(path) for path in full_file_paths_Pa]
print(f"Successfully loaded {len(pa_data_list)} air pressure files.")
```
</details>

### 1.4 Concatenating and Merging All Data
Now that we have all our data loaded, we need to combine it into one master DataFrame for analysis. This involves two steps:
Concatenate: Stacking the files of the same type together (e.g., all gas files into one, all temperature files into one).
Merge: Joining the different datasets (gas, temperature, and pressure) together based on their common timestamp.

**Concatenating the Datasets**

First, let's use pd.concat() to combine the lists of DataFrames we created. After combining, we must format the Timestamp column and set it as the index, just as we did before.

```Python

# --- Concatenate and Clean Gas Data ---
df_gas = pd.concat(raw_data_list) # Assumes raw_data_list is from the previous step

# --- Concatenate and Clean Temperature Data ---
df_Ta = pd.concat(ta_data_list)
df_Ta['Timestamp'] = pd.to_datetime(df_Ta['Timestamp'])
df_Ta = df_Ta.set_index('Timestamp')

# --- Concatenate and Clean Pressure Data ---
df_Pa = pd.concat(pa_data_list)
df_Pa['Timestamp'] = pd.to_datetime(df_Pa['Timestamp'])
df_Pa = df_Pa.set_index('Timestamp')

print("--- Gas DataFrame Info ---")
df_gas.info()
print("\n--- Temperature DataFrame Info ---")
df_Ta.info()
print("\n--- Pressure DataFrame Info ---")
df_Pa.info()

```
**Merging Gas and Auxiliary Data**

Finally, we need to combine our df_gas, df_Ta, and df_Pa DataFrames. We want to add the temperature and pressure columns to the gas data, matching them by the nearest timestamp.
The gas analyzer records data every second, while the weather station might only record every minute. A simple merge would leave many empty rows. The perfect tool for this is pd.merge_asof(). It performs a "nearest-neighbor" merge, which is ideal for combining time-series data with different frequencies.

```Python

# First, merge the two auxiliary datasets together
df_aux = pd.merge_asof(left=df_Ta, right=df_Pa, on='Timestamp', direction='nearest')

# Now, merge the gas data with the combined auxiliary data.
# We use direction='backward' to find the most recent weather data for each gas measurement.
df_raw = pd.merge_asof(
    left=df_gas, 
    right=df_aux, 
    on='Timestamp', 
    direction='backward'
)

print("\n--- Final Merged DataFrame ---")
display(df_raw.head())
df_raw.info()

```

Brilliant! You now have a single, clean DataFrame called df_final that contains everything you need: the high-frequency gas concentrations and the corresponding temperature and pressure for each measurement point. We are now fully prepared to move on to the flux calculation.

## 2. Visualizing and Cleaning the Data
Now that we have a single, merged DataFrame, our next step is to inspect the data quality. Raw sensor data from the field is almost never perfect. Visualizing it is the best way to diagnose issues like noise, drift, or outliers before we attempt any calculations. For this, we'll use plotly, a powerful library for creating interactive plots.
### 2.1 Creating a Reusable Plotting Function with Plotly
Just as we did with data loading, we'll be plotting our time-series data multiple times. To make this efficient and keep our plots looking consistent, let's create a dedicated function. This function will take a DataFrame and some plot details as input and generate an interactive plot.
> **Task**: now create the reusable plotting function!

```python

def plot_time_series_plotly(df, y_column, title, mode='lines'):
    """
    Generates an interactive time-series plot using Plotly.

    Parameters:
    - df (pd.DataFrame): DataFrame with a DatetimeIndex.
    - y_column (str): The name of the column to plot on the y-axis.
    - title (str): The title for the plot.
    - mode (str): Plotly mode ('lines', 'markers', or 'lines+markers').
    """
    fig = go.Figure()

    fig.add_trace(...)

    # Update layout for a clean look
    fig.update_layout(
        ...
    )
    
    fig.show()
```

<details>
    <summary>Here is the solution!</summary>
        
    ```Python
    
    import plotly.graph_objects as go
    import plotly.io as pio
    
    # This setting forces Plotly to open plots in your default web browser,
    # which can be more stable in some environments.
    pio.renderers.default = "browser"
    
    def plot_time_series_plotly(df, y_column, title, mode='lines'):
        """
        Generates an interactive time-series plot using Plotly.
        This function will automatically try to set a 'Timestamp' column as the 
        index if the existing index is not a datetime type.
    
        Parameters:
        - df (pd.DataFrame): DataFrame to plot.
        - y_column (str): The name of the column to plot on the y-axis.
        - title (str): The title for the plot.
        - mode (str): Plotly mode ('lines', 'markers', or 'lines+markers').
        """
        # --- Input Validation and Auto-Correction ---
        
        # It's good practice to work on a copy inside a function to avoid 
        # changing the user's original DataFrame unexpectedly.
        df_plot = df.copy()
    
        if not pd.api.types.is_datetime64_any_dtype(df_plot.index):
            print("Note: The DataFrame index is not a DatetimeIndex.")
            # Attempt to fix the issue by finding a 'Timestamp' column
            if 'Timestamp' in df_plot.columns:
                print("--> Found a 'Timestamp' column. Attempting to set it as the index.")
                df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
                # CRITICAL: You must re-assign the variable to save the change.
                df_plot = df_plot.set_index('Timestamp')
            else:
                # If we can't fix it automatically, then we raise an error.
                raise TypeError(
                    "The DataFrame index is not a DatetimeIndex and a 'Timestamp' column was not found. "
                    "Please set a DatetimeIndex before plotting."
                )
                
        # --- Plotting ---
        # By this point, df_plot is guaranteed to have a valid DatetimeIndex.
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=df_plot.index, 
            y=df_plot[y_column], 
            mode=mode, 
            name=y_column
        ))
    
        # Update layout for a clean, professional look
        fig.update_layout(
            title=title, 
            xaxis_title='Time', 
            yaxis_title=f'{y_column} Concentration (ppb)', 
            template='plotly_white', 
            title_font=dict(size=24),
            xaxis=dict(tickfont=dict(size=14), title_font=dict(size=16)), 
            yaxis=dict(tickfont=dict(size=14), title_font=dict(size=16))
        )
        
        fig.show()
    
    ```
</details>


### 2.2 Visualizing the Raw Gas Data
Now, let's use our new function to look at the raw N₂O data from our combined file. You can zoom and pan on the plot to inspect the noisy areas.

```Python

# Call our function to plot the raw 'N2O' column
plot_time_series_plotly(df_final, y_column='N2O', title='Raw N2O Concentration Over Time')

```

![raw data plotting](/assets/images/python/5/raw_data_plot.png)

As you can see from the plot, the raw data is very noisy. There are several negative values and some extremely large spikes. These are physically impossible and are likely due to sensor errors or electrical interference. We cannot calculate meaningful fluxes from this data without cleaning it first.

### 2.3 Filtering with a Quantile Filter
To remove these outliers, we'll use a simple but effective quantile filter. This method is robust because the extreme values we want to remove have very little influence on the calculation of percentiles. We will calculate the 10th and 90th percentiles of the N₂O concentration and discard any data points that fall outside this range.

```Python

# Calculate the 10th and 90th percentiles
p_10 = df_final.N2O.quantile(0.10)
p_90 = df_final.N2O.quantile(0.90)

print(f"Filtering data to keep N2O concentrations between {p_10:.2f} and {p_90:.2f} ppb.")

# Apply the filter to create a new, clean DataFrame
# .copy() is used here to avoid a SettingWithCopyWarning from pandas
df_filtered = df_final[(df_final.N2O >= p_10) & (df_final.N2O <= p_90)].copy()

# Visualize the filtered data using our function again, this time using 'markers'
plot_time_series_plotly(df_filtered, y_column='N2O', title='Filtered N2O Concentration Over Time', mode='markers')
```

![Filtered N2O](/assets/images/python/5/filtered_N2O.png)


This looks much better! The noise is gone, and a clear, meaningful pattern has emerged.

### 2.4 Understanding the Data Pattern
The filtered data shows a repeating pattern which is the signature of the static chamber method:
Baseline (Ambient Air): The long, relatively flat periods show the baseline N₂O concentration in the ambient air.
Concentration Increase (Chamber Closed): The sections where the concentration rises steadily and linearly are the actual measurements. This occurs when the chamber is placed over the soil, trapping the gases being emitted. The rate of this increase is what we will use to calculate the flux.
Sudden Drop (Chamber Opened): The sharp vertical drops occur when a measurement is finished, and the chamber is lifted from the ground, exposing the sensor to ambient air again.
Leveling Off: If a chamber is left on the ground for too long, the gas concentration inside can build up, altering the pressure gradient between the soil and the chamber air. This can cause the rate of increase to slow down and "level off." For this reason, it's crucial to use only the initial, linear part of the increase for our flux calculation.

## 3. Calculating Flux for a Single Measurement
After loading and filtering our raw data and getting an overview of the patterns, it's time to calculate the fluxes. Excited?
In this section, we will focus on the data for a single measurement period to understand the process in detail. We'll break it down into a few key steps:
<ol>
<li> Review the flux calculation formula to see what components we need. </li>
<li> Define the metadata (chamber dimensions, etc.) for our specific plot. </li>
<li> Isolate the data for a specific time window and visualize it. </li>
<li> Perform a linear regression on the concentration data to get the rate of change. </li>
<li> Combine all the pieces to calculate the final flux. </li>
</ol>

### 3.1 The Flux Calculation Formula

The flux is calculated based on the Ideal Gas Law (PV = nRT). It tells us how much gas (n) is in a given volume (V) at a specific pressure (P) and temperature (T). By measuring how fast the amount of gas increases, we can determine the flux from the soil.
The formula of flux calculation looks like this:

​
$$
\text{Flux Rate (molar)} = \frac{\frac{\Delta C}{t} \cdot V \cdot p}{R \cdot (T_{c} + 273.15) \cdot A}
$$
​

Where:
ΔC/t: The rate of change of the gas concentration in ppm/s (this is the slope from our regression).
V: The total volume of the chamber headspace (m³).
p: The air pressure in Pascals (Pa) during measurement.
R: The ideal gas constant (8.314 J K⁻¹ mol⁻¹).
T_c: The air temperature in Celsius (°C).
A: The surface area covered by the chamber (m²).

To better understand the above fomula, it can be arranged into the following:


$$
\text{Flux Rate (molar)} = \left( \frac{\Delta C}{t} \right) \cdot \left( \frac{p}{R \cdot (T_C + 273.15)} \right) \cdot \left( \frac{V}{A} \right)
$$


Now, it is clear that the fomula only contains three components: **Flux** = **slope** * **molar_density** * **V_over_A**
Okay, lets create a function of flux calculation based on the fomula for later use.



<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">

{% capture exercise %}
### Exercise

The function `calculate_flux` is provided below but is not complete. It is your task to finish the function based on the formula.

```python
# Define key physical constants
R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)

def calculate_flux(slope_ppb_s, temp_k, pressure_pa, v_over_a):
    """
    Calculates N2O flux.
    """
    # Convert slope from ppb/s to ppm/s for the formula
    ppm_per_second = ...
    
    # Calculate molar density of air (n/V = P/RT) in mol/m³
    molar_density = ...
    
    # Calculate the flux in µmol m⁻² s⁻¹
    # The 1e6 converts from mol to µmol
    flux = ...
    return flux

<details>
    <summary>Solution!</summary>
    Here is the completed function:
    
    ```python
    # Define key physical constants
    R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)
    
    def calculate_flux(slope_ppb_s, temp_k, pressure_pa, v_over_a):
        """
        Calculates N2O flux.
        """
        # Convert slope from ppb/s to ppm/s for the formula
        ppm_per_second = slope_ppb_s / 1000.0
        
        # Calculate molar density of air (n/V = P/RT) in mol/m³
        molar_density = pressure_pa / (R * temp_k)
        
        # Calculate the flux in µmol m⁻² s⁻¹
        # The 1e6 converts from mol to µmol
        flux = ppm_per_second * molar_density * v_over_a * 1e6
        return flux
    ```
</details>

{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>


3.2 Defining Measurement Metadata
Now, we need the physical dimensions of our chamber setup for a specific plot. This information comes from our "digital field notebook." For this example, let's define the metadata for a single plot.
code
Python
# Metadata for a single example measurement
plot_metadata = pd.Series({
    'plot_id': 1,
    'land_use': 'forest',
    'collar_height_m': 0.055,  # Height of the collar ring in the ground
    'chamber_height_m': 0.40,  # Height of the chamber itself
    'chamber_radius_m': 0.2,   # Radius of the chamber
})

# Calculate the Volume-to-Area (V/A) ratio from the metadata
chamber_area = np.pi * plot_metadata['chamber_radius_m']**2
total_height = plot_metadata['collar_height_m'] + plot_metadata['chamber_height_m']
chamber_volume = total_height * chamber_area
v_over_a = chamber_volume / chamber_area

print(f"The Volume-to-Area (V/A) ratio for this plot is: {v_over_a:.3f} m")
3.3 Isolating and Visualizing the Measurement Data
Let's use your example time period: 2025-08-15 12:00:00 to 2025-08-15 12:10:00. We'll slice our df_filtered DataFrame to get only the data within this window and then plot it to get a closer look.
code
Python
# Define the start and end times for our measurement window
start_time = '2025-08-15 12:04:00'
end_time = '2025-08-15 12:09:30'

# Select the data for this specific time window
measurement_data = df_filtered[start_time:end_time]

# Use our plotting function to visualize this specific period
plot_time_series_plotly(
    measurement_data, 
    y_column='N2O', 
    title=f'N2O Concentration for Plot {plot_metadata["plot_id"]}',
    mode='markers'
)
This plot shows the clear, linear increase in N₂O concentration after the chamber was placed on the collar. This is the exact data we need for our regression.
3.4 Linear Regression to Find the Slope
Now for the most important part of the analysis. We will fit a straight line to these data points. The slope of that line is the dC/dt (rate of change) that we need for our flux formula.
For the regression, our x-axis needs to be a simple number (like seconds elapsed), not a timestamp. So, our first step is to create a new column, elapsed_seconds.
code
Python
from scipy import stats

# Make a copy to avoid a SettingWithCopyWarning
measurement_data = measurement_data.copy()

# Create an 'elapsed_seconds' column for the regression
start_timestamp = measurement_data.index.min()
measurement_data['elapsed_seconds'] = (measurement_data.index - start_timestamp).total_seconds()

# Perform the linear regression using SciPy
slope, intercept, r_value, p_value, std_err = stats.linregress(
    x=measurement_data['elapsed_seconds'],
    y=measurement_data['N2O']
)

# The R-squared value tells us how well the line fits the data (a value > 0.9 is good!)
r_squared = r_value**2

print(f"--- Regression Results ---")
print(f"Slope (dC/dt): {slope:.4f} ppb/s")
print(f"R-squared: {r_squared:.4f}")
3.5 Visualizing the Fit and Final Calculation
It's always good practice to visualize the regression line against the data to confirm the fit is good.
code
Python
# --- Visualize the regression line ---
fig = go.Figure()

# Add the raw data points
fig.add_trace(go.Scatter(x=measurement_data['elapsed_seconds'], y=measurement_data['N2O'], 
                         mode='markers', name='Raw Data'))

# Add the fitted regression line
fig.add_trace(go.Scatter(x=measurement_data['elapsed_seconds'], 
                         y=intercept + slope * measurement_data['elapsed_seconds'],
                         mode='lines', name='Fitted Line', line=dict(color='red')))

fig.update_layout(title=f'Linear Regression for Plot {plot_metadata["plot_id"]} (R²={r_squared:.2f})',
                  xaxis_title='Elapsed Time (s)', yaxis_title='N2O Concentration (ppb)', template='plotly_white')
fig.show()

# --- Finally, Calculate the Flux! ---
# Get the average temperature and pressure during the measurement
avg_temp_c = measurement_data['T_air'].mean()
avg_pressure_pa = measurement_data['P_air'].mean() * 100 # Assuming pressure is in hPa, convert to Pa

# Convert temperature to Kelvin
avg_temp_k = avg_temp_c + 273.15

# Now we have all the pieces! Let's call our function.
final_flux = calculate_flux(
    slope_ppb_s=slope,
    temp_k=avg_temp_k,
    pressure_pa=avg_pressure_pa,
    v_over_a=v_over_a
)

print(f"\n--- Final Flux Calculation ---")
print(f"Average Temperature: {avg_temp_c:.2f} °C")
print(f"Average Pressure: {avg_pressure_pa:.2f} Pa")
print(f"Calculated N₂O Flux: {final_flux:.5f} µmol m⁻² s⁻¹")



rate to calculate a flux, we need to combine the gas concentration data with **metadata** about our measurement setup. This includes the start and end times of each chamber placement and the physical dimensions of the chamber (like collar height).

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
