---
title: "4. FLUX CALCULATION"
layout: default
nav_order: 5
---

# **Flux Calculation**
In this tutorial, we're going to analyze the data you collected on your field trip to the Lüner forest! Your instruments measured raw gas concentrations, but as ecologists, we need to turn that into gas fluxes. Why? Because fluxes represent a rate—the speed at which gases are being exchanged. With CO₂ fluxes, we can estimate crucial metrics like ecosystem respiration (RECO) and net ecosystem exchange (NEE). With fluxes of a potent greenhouse gas like Nitrous Oxide (N₂O), we can understand a key part of the nitrogen cycle. This guide will walk you through the entire process: from cleaning the raw concentration data, to calculating meaningful fluxes, and finally to comparing the results between different land cover types.
> **Notice:**
>In the following sections, we will start using new functions and libraries that we haven't introduced yet. Don't worry or feel overwhelmed! This is a normal part of learning to code. For each new tool we use, I will:     Briefly explain what it is and why we are using it. Provide a link to its official documentation if you're curious and want to learn more. Think of it as adding new tools to your data analysis toolbox. We'll introduce them one at a time, right when we need them.


### Table of Contents
1. [Read in and merge data files](#1-read-in-and-merge-data-files)
2. [Visualizing and cleaning the data](#2-visualizing-and-cleaning-the-data)
3. [Calculating flux for a single measurement](#3-calculating-flux-for-a-single-measurement)
4. [Automating gas flux calculation](#4-automating-gas-flux-calculation)



## 1.Read in and merge data files
Different from the simple CSV files we might have worked with before, the raw data from the gas analyzer is more complex. When you open the file, you'll see it contains two parts:
A metadata header: This block at the top contains useful information about the measurement (like timezone, device model, etc.), but we don't need it for our flux calculations.
The data block: This is the core data we need, with columns for date, time, and gas concentrations.
Our first challenge is to programmatically read only the data block and ignore the metadata.

```
Model:	LI-7820
SN:	TG20-01072
Software Version:	2.3.8
Timestamp:	2025-08-15 11:00:00
Timezone:	Europe/Paris
```

To do this, we'll need the pandas library for creating our DataFrame and the io library, we need to import them.
```python
import pandas as pd
import io
```
    
Our strategy will be to read the file line-by-line, find the start of the data, and then pass only those lines to pandas.
### 1.1 Loading N₂O Data
The analyzer produces tab-separated files with a metadata block at the top. The data section is marked by a line starting with DATAH. Our strategy is to read the file line-by-line, find the DATAH marker, and pass only the data lines to pandas.

### Reading and parsing the file
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
### Using io.StringIO to Read Our Cleaned Data
The pd.read_csv() function is built to read from a file. We don't have a clean file; we have a list of Python strings (lines) that we've already processed.
So, how do we make pandas read from our list? We use io.StringIO to trick pandas. It takes our cleaned-up data lines and presents them to pandas as if they were a file stored in the computer's memory.
>**Info**:
>The Python io module helps us manage data streams. io.StringIO specifically allows us to treat a regular text string as a >file. This is incredibly useful when you need to pass text data to a function that expects a file, just like we're doing >with pd.read_csv().

```python
# Join our data lines back into a single string, separated by newlines
data_string = '\n'.join(lines[data_start_index:])
# Read the data string into a DataFrame
df_n2o = pd.read_csv(
    io.StringIO(data_string),  # Treat our string as a file
    sep='\t',                  # Tell pandas the data is separated by tabs
    header=None,               # We are providing the headers ourselves, so there isn't one in the data
    names=headers,             # Use the 'headers' list we extracted earlier
    na_values='nan'            # Recognize 'nan' strings as missing values
)
```

### Data Formatting
The last step is to tidy up the DataFrame. We will:
Remove the useless DATAH column.
Combine the separate DATE and TIME columns into a single Timestamp object. This is crucial for time-series analysis.
Set this new Timestamp as the DataFrame's index, which makes plotting and selecting data by time much easier.

```python
# Drop the first column which is just the 'DATAH' label
if 'DATAH' in df_n2o.columns:
    df_n2o = df_n2o.drop(columns=['DATAH'])

# Combine 'DATE' and 'TIME' into a proper Timestamp and set it as the index
if 'DATE' in df_n2o.columns and 'TIME' in df_n2o.columns:
    df_n2o['Timestamp'] = pd.to_datetime(df_n2o['DATE'] + ' ' + df_n2o['TIME'])
    df_n2o = df_n2o.drop(columns=['DATE', 'TIME'])
    df_n2o = df_n2o.set_index('Timestamp')

print("Data loaded and formatted successfully!")
print(df_n2o.head())
```
Great! Now, we have successfully read in and formatted our raw data.
However, think about our field campaigns. We went out several times and generate a new data file for each trip. If we wanted to analyze all of them, we would have to copy and paste our loading code multiple times.
To avoid repetition and make our code cleaner and more reliable, it's a best practice to wrap a reusable process into a function. Let's turn our loading and cleaning steps into a function called load_n2o_data.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">

{% capture exercise %}
### Exercise

Try to write this function yourself based on the code snippets we created for data loading!
Tip: The function will need to accept one argument: the filepath of the file you want to open.

<details markdown="1"><summary>Solution!</summary>
Note: how it's the exact same logic as before, just defined within a def block.
    
```python
def load_n2o_data(filepath: str) -> pd.DataFrame:
    """
    Loads N2O data from a text file, remove metadata, and returns a DataFrame.

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

    df_n2o = pd.read_csv(
        io.StringIO('\n'.join(lines[data_start_index:])),
        sep='\t',
        header=None,
        names=headers,
        na_values='nan'
    )

    if 'DATAH' in df_n2o.columns:
        df_n2o = df_n2o.drop(columns=['DATAH'])

    if 'DATE' in df_n2o.columns and 'TIME' in df_n2o.columns:
        df_n2o['Timestamp'] = pd.to_datetime(df_n2o['DATE'] + ' ' + df_n2o['TIME'])
        df_n2o = df_n2o.drop(columns=['DATE', 'TIME'])
        df_n2o = df_n2o.set_index('Timestamp')

    print("N2O data loaded and cleaned successfully.")
    return df_n2o
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

### 1.2 Loading CH₄ and CO₂ Data

The GGA analyzer produces comma-separated files with a different structure:
- **Line 1:** Instrument metadata (version, serial number, etc.)
- **Line 2:** Column headers
- **Lines 3+:** Measurement data

Here's an example of the first few lines:

```
VC:2f90039 BD:Jan 16 2014 SN:
                     Time,      [CH4]_ppm,   [CH4]_ppm_sd,      [H2O]_ppm, ...
  08/15/2025 11:00:03.747,   2.080375e+00,   0.000000e+00,   1.103072e+03, ...
```

However, there's a complication: some GGA files contain extra non-data content at the end (such as digital signatures or log messages). We need to filter these out. Let's build our loader step by step.

### Read the CSV File

First, we read the file with `pd.read_csv()`, skipping the first metadata line:

```python
df_gga = pd.read_csv(
    "./BAI_StudyProject_LuentenerWald/raw_data/gga_2025-08-15_f0000.txt",
    skiprows=1,            # Skip instrument metadata header (line 1)
    skipinitialspace=True  # Handle leading whitespace in columns
)

# Clean column names (remove extra spaces)
df_gga.columns = df_gga.columns.str.strip()

print(f"Rows loaded: {len(df_gga)}")
df_gga.head()
```

### Identify Valid Data Rows

If we look at the end of some files, we might find non-data content like this:

```
-----BEGIN PGP MESSAGE-----
Version: GnuPG v1.4.11 (GNU/Linux)
jA0EAwMC1o7j8zNG6eRgye1CgI1h0/yQoOa8fycg+...
```

We need to keep only the rows where the `Time` column contains an actual timestamp. Valid timestamps in our data look like `08/15/2025 11:00:03.747` — they always start with a date in `MM/DD/YYYY` format.

To identify these rows programmatically, we use a **regular expression** (regex). A regex is a pattern that describes what text should look like.

> **Info: What is a Regular Expression?**
> 
> A regular expression is a sequence of characters that defines a search pattern. It's like a template that says "I'm looking for text that looks like THIS." Regular expressions are extremely powerful for finding, matching, and filtering text data.

Here's the regex pattern we'll use: `^\s*\d{2}/\d{2}/\d{4}`

Let's break it down piece by piece:

| Pattern | Meaning | Example Match |
|---------|---------|---------------|
| `^` | Start of the string | (anchors the match to the beginning) |
| `\s*` | Zero or more whitespace characters | Matches leading spaces like `"  "` |
| `\d{2}` | Exactly 2 digits | `08` (month) |
| `/` | A literal forward slash | `/` |
| `\d{2}` | Exactly 2 digits | `15` (day) |
| `/` | A literal forward slash | `/` |
| `\d{4}` | Exactly 4 digits | `2025` (year) |

So the full pattern `^\s*\d{2}/\d{2}/\d{4}` means: "Starting from the beginning, allow optional spaces, then match a date in MM/DD/YYYY format."
> **Learn More About Regular Expressions**
> 
> Regular expressions are a powerful tool worth learning. Here are some helpful resources:
> - [RegExr](https://regexr.com/) — Interactive regex tester with real-time explanations
> - [Regex101](https://regex101.com/) — Another great tester with detailed breakdown of patterns
> - [Python re documentation](https://docs.python.org/3/library/re.html) — Official Python regex documentation
> - [Regular Expressions Cheat Sheet](https://www.dataquest.io/blog/regex-cheatsheet/) — Quick reference for common patterns

Let's see it in action:

```python
import re

# Test the pattern on different strings
pattern = r'^\s*\d{2}/\d{2}/\d{4}'

test_strings = [
    '  08/15/2025 11:00:03.747',    # Valid timestamp (with leading spaces)
    '08/15/2025 11:00:03.747',       # Valid timestamp (no leading spaces)
    '-----BEGIN PGP MESSAGE-----',   # Invalid (PGP signature)
    'jA0EAwMC1o7j8zNG6eRgye1C',      # Invalid (encrypted data)
    'nan',                            # Invalid (missing value)
]

for s in test_strings:
    match = bool(re.match(pattern, s))
    print(f"'{s[:30]:30s}' → {match}")
```

Output:
```
'  08/15/2025 11:00:03.747     ' → True
'08/15/2025 11:00:03.747       ' → True
'-----BEGIN PGP MESSAGE-----  ' → False
'jA0EAwMC1o7j8zNG6eRgye1C      ' → False
'nan                           ' → False
```

Now we can use this pattern to filter our DataFrame:

```python
# Create a boolean mask: True for valid rows, False for invalid
valid_mask = df_gga['Time'].astype(str).str.match(r'^\s*\d{2}/\d{2}/\d{4}')

# Count how many rows we're keeping vs. dropping
print(f"Valid rows: {valid_mask.sum()}")
print(f"Invalid rows (will be dropped): {(~valid_mask).sum()}")

# Keep only valid rows
df_gga = df_gga[valid_mask].copy()
```

### Parse Timestamps

Now that we have only valid data, we convert the `Time` column to proper datetime objects:

```python
# Remove any leading/trailing whitespace and parse the timestamp
df_gga['Time'] = pd.to_datetime(
    df_gga['Time'].str.strip(), 
    format='%m/%d/%Y %H:%M:%S.%f'
)

# Rename to 'Timestamp' for consistency and set as index
df_gga = df_gga.rename(columns={'Time': 'Timestamp'})
df_gga = df_gga.set_index('Timestamp')

print(f"Time range: {df_gga.index.min()} to {df_gga.index.max()}")

print("check the last five rows of the table to see if we really removed the non-data part")
df_gga.tail()
```

### Putting It All Together

Now let's wrap everything into a reusable function:

```python
def load_gga_data(filepath: str) -> pd.DataFrame:
    """
    Load CH4/CO2 data from a Los Gatos GGA analyzer file.
    
    Parameters:
        filepath: Path to the GGA .txt file
        
    Returns:
        DataFrame with DatetimeIndex
    """
    # Step 1: Read CSV, skip metadata header
    df_gga = pd.read_csv(
        filepath,
        skiprows=1,
        skipinitialspace=True
    )
    df_gga.columns = df_gga.columns.str.strip()
    
    # Step 2: Filter to valid data rows using regex
    # Pattern: optional whitespace, then MM/DD/YYYY date format
    valid_mask = df_gga['Time'].astype(str).str.match(r'^\s*\d{2}/\d{2}/\d{4}')
    df_gga = df_gga[valid_mask].copy()
    
    # Step 3: Parse timestamps and set as index
    df_gga['Time'] = pd.to_datetime(df_gga['Time'].str.strip(), format='%m/%d/%Y %H:%M:%S.%f')
    df_gga = df_gga.rename(columns={'Time': 'Timestamp'})
    df_gga = df_gga.set_index('Timestamp')
    
    return df_gga
```

Let's test it:

```python
df_gga = load_gga_data("./BAI_StudyProject_LuentenerWald/raw_data/gga_2025-08-15_f0000.txt")
print(f"Loaded {len(df_gga):,} rows")
print(f"Time range: {df_gga.index.min()} to {df_gga.index.max()}")
df_gga.head()
```



### 1.3 Loading Multiple Files
Now that we have loader functions, we can easily handle data from multiple field trips. Instead of copying code, we can simply call our function in a loop.
First, we create a list of all the file paths we want to load. Then, we can loop through this list, call our function for each path, and store the resulting DataFrames in a new list.

```python
# First, let's list all the files we want to load.
# Make sure the file paths are complete and correct.
base_path = "./BAI_StudyProject_LuentenerWald/raw_data/"
n2o_files = [
    'TG20-01072-2025-08-15T110000.data.txt',
    'TG20-01072-2025-08-26T093000.data.txt'
]

gga_files = [
    'gga_2025-08-15_f0000.txt',
    'gga_2025-08-06_f0000.txt',
    'gga_2025-08-26_f0000.txt'
]

# Create the full file paths
n2o_full_file_paths = [base_path + name for name in n2o_files]
gga_full_file_paths = [base_path + name for name in gga_files]

# Create an empty list to hold the loaded DataFrames
n2o_data_list = []
gga_data_list = []

# Loop through each path, load the data, and append it to our list
for path in n2o_full_file_paths:
    df = load_n2o_data(path)
    n2o_data_list.append(df)

print(f"\nSuccessfully loaded {len(n2o_data_list)} N2O data files.")

for path in gga_full_file_paths:
    df = load_gga_data(path)
    gga_data_list.append(df)

print(f"\nSuccessfully loaded {len(gga_data_list)} GGA data files.")
```

The loop above is clear and correct. However, a more concise way to write this in Python is with a list comprehension. It achieves the exact same result in a single, readable line:

```python
n2o_data_list = [load_n2o_data(path) for path in n2o_full_file_paths]
gga_data_list = [load_gga_data(path) for path in gga_full_file_paths]
```

For our flux calculations to be accurate, we need more than just gas concentrations. The Ideal Gas Law, which is the basis of the calculation, requires the temperature and air pressure at the time of each measurement.
We will use the same workflow as before: load each file and then combine them.


<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">

{% capture exercise %}
### Exercise

You have two Excel files containing air temperature.
Create lists of the file paths for the temperature data.
Load each Excel file into a pandas DataFrame. Try using a list comprehension as we learned before!

<details markdown="1">
    
<summary>Click here for the solution!</summary>
    
```python
# the base path is the same as before
base_path = "./BAI_StudyProject_LuentenerWald/raw_data/"

# --- Load Air Temperature Data ---
file_names_Ta = [
    'Haube 2025-08-06 15_24_26 CEST (Data CEST).xlsx', 
    'Haube 2025-08-15 15_22_46 CEST (Data CEST).xlsx' ,
    'Haube 2025-08-26 14_58_20 MESZ (Data MESZ).xlsx'
]
full_file_paths_Ta = [base_path + name for name in file_names_Ta]
ta_data_list = [pd.read_excel(path) for path in full_file_paths_Ta]
print(f"Successfully loaded {len(ta_data_list)} air temperature files.")
```

</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>



### 1.4 Concatenating and Merging All Data

Now we combine everything into one master DataFrame. Understanding our data structure is important:

- **Gas measurements (N₂O, CH₄, CO₂):** Recorded at different times. They do NOT overlap in time, we simply need to stack them together.
- **Temperature data:** Recorded continuously and DOES overlap with all gas measurements. We need to match each gas reading with its corresponding temperature.

This means our workflow is:

1. **Concatenate** all gas data files into one DataFrame (stacking rows)
2. **Merge** temperature into the gas data using time-matching

### Concatenate Gas Data from All Files

First, let's combine files of the same type:

```python
# Concatenate N2O data from multiple files
df_n2o = pd.concat(n2o_data_list)
df_n2o = df_n2o.sort_index()

# Concatenate GGA data (CH4 and CO2) from multiple files
df_gga = pd.concat(gga_data_list)
df_gga = df_gga.sort_index()

# Concatenate and format temperature data
df_Ta = pd.concat(ta_data_list)
df_Ta['Timestamp'] = pd.to_datetime(df_Ta['Date-Time (CEST)'])
df_Ta = df_Ta.set_index('Timestamp')
df_Ta = df_Ta.sort_index()

print("--- N2O DataFrame ---")
print(f"  Rows: {len(df_n2o):,}, Time range: {df_n2o.index.min()} to {df_n2o.index.max()}")

print("\n--- GGA DataFrame (CH4/CO2) ---")
print(f"  Rows: {len(df_gga):,}, Time range: {df_gga.index.min()} to {df_gga.index.max()}")

print("\n--- Temperature DataFrame ---")
print(f"  Rows: {len(df_Ta):,}, Time range: {df_Ta.index.min()} to {df_Ta.index.max()}")
```

### Combine All Gas Data into One Master Table

Since N₂O and CH₄/CO₂ measurements don't overlap in time, we can safely stack them together. First, we need to select and rename columns so they're consistent:

```python
# Select key columns from N2O data
df_n2o_clean = df_n2o[['N2O']].copy()
df_n2o_clean.columns = ['N2O_ppb']

# Select key columns from GGA data (dry-corrected values)
df_gga_clean = df_gga[['[CH4]d_ppm', '[CO2]d_ppm']].copy()
df_gga_clean.columns = ['CH4_ppm', 'CO2_ppm']

# Stack them together - rows from different time periods
df_gas = pd.concat([df_n2o_clean, df_gga_clean])
df_gas = df_gas.sort_index()

print(f"Combined gas DataFrame: {len(df_gas):,} rows")
print(f"Time range: {df_gas.index.min()} to {df_gas.index.max()}")
df_gas.head()
```

Notice that each row will have values in either `N2O_ppb` OR `CH4_ppm`/`CO2_ppm`, but not both—because the measurements were taken at different times:

```python
# Check the structure
print("\nSample from N2O measurement period:")
print(df_gas.loc[df_gas['N2O_ppb'].notna()].head(3))

print("\nSample from CH4/CO2 measurement period:")
print(df_gas.loc[df_gas['CH4_ppm'].notna()].head(3))
```

### Merge Temperature with Gas Data

The gas analyzers record data every second, while the weather station might record only every minute. A simple merge would leave many empty rows. The solution is `pd.merge_asof()`, which performs a "nearest-neighbor" merge—ideal for combining time-series data with different frequencies.

```python
# Reset index for merge_asof (requires sorted column, not index)
df_gas_reset = df_gas.reset_index()
df_Ta_reset = df_Ta.reset_index()

# Merge gas data with temperature
# direction='backward' means: for each gas reading, find the most recent temperature
df_merged = pd.merge_asof(
    df_gas_reset.sort_values('Timestamp'),
    df_Ta_reset[['Timestamp', 'Temperature , °C']].sort_values('Timestamp'),
    on='Timestamp',
    direction='backward'
)

print(f"Merged DataFrame: {len(df_merged):,} rows")
df_merged.head()
```

### Final Master DataFrame

Let's verify our final dataset:

```python
print("--- Master DataFrame ---")
print(f"Total rows: {len(df_merged):,}")
print(f"Time range: {df_merged.index.min()} to {df_merged.index.max()}")
print(f"\nColumns: {list(df_merged.columns)}")

print("\n--- Sample rows with N2O data ---")

from IPython.display import display
display(df_merged.loc[df_merged['N2O_ppb'].notna()].head(100))

print("\n--- Sample rows with CH4/CO2 data ---")
display(df_merged.loc[df_merged['CH4_ppm'].notna()].head(100))
```

The master DataFrame now contains:
- `N2O_ppb`: N₂O concentration (only during N₂O measurement periods)
- `CH4_ppm`: CH₄ concentration (only during GGA measurement periods)  
- `CO2_ppm`: CO₂ concentration (only during GGA measurement periods)
- `Ta_C`: Air temperature (matched to each gas reading)

  
## 2. Visualizing and cleaning the data
Now that we have a single, merged DataFrame, our next step is to inspect the data quality. Raw sensor data from the field is almost never perfect. Visualizing it is the best way to diagnose issues like noise, drift, or outliers before we attempt any calculations. For this, we'll use plotly, a powerful library for creating interactive plots.
### 2.1 Creating a Reusable Plotting Function with Plotly
Just as we did with data loading, we'll be plotting our time-series data multiple times. To make this efficient and keep our plots looking consistent, let's create a dedicated function. This function will take a DataFrame and some plot details as input and generate an interactive plot.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise
The plooting function is partly providing in the following, now finish the function!

```python

def plot_time_series(df, y_column, title, mode='lines'):
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

<details markdown="1">
<summary>Here is the solution!</summary>
    
```python
import plotly.graph_objects as go
import plotly.io as pio

# This setting forces Plotly to open plots in your default web browser,
# which can be more stable in some environments.
pio.renderers.default = "browser"

def plot_time_series(df, y_column, title, mode='lines'):
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
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

### 2.2 Visualizing the Raw Gas Data
Now, let's use our new function to look at the raw N₂O data from our combined file. You can zoom and pan on the plot to inspect the noisy areas.

```python

# Call our function to plot the raw 'N2O' column
plot_time_series(df_final, y_column='N2O', title='Raw N2O Concentration Over Time')

```

![raw data plotting](/assets/images/python/5/raw_data_plot.png)

As you can see from the plot, the raw data is very noisy. There are several negative values and some extremely large spikes. These are physically impossible and are likely due to sensor errors or electrical interference. We cannot calculate meaningful fluxes from this data without cleaning it first.

### 2.3 Filtering with a Quantile Filter
To remove these outliers, we'll use a simple but effective quantile filter. This method is robust because the extreme values we want to remove have very little influence on the calculation of percentiles. We will calculate the 10th and 90th percentiles of the N₂O concentration and discard any data points that fall outside this range.

```python
# Calculate the 10th and 90th percentiles
p_10 = df_final.N2O.quantile(0.10)
p_90 = df_final.N2O.quantile(0.90)

print(f"Filtering data to keep N2O concentrations between {p_10:.2f} and {p_90:.2f} ppb.")

# Apply the filter to create a new, clean DataFrame
# .copy() is used here to avoid a SettingWithCopyWarning from pandas
df_filtered = df_final[(df_final.N2O >= p_10) & (df_final.N2O <= p_90)].copy()

# Visualize the filtered data using our function again, this time using 'markers'
plot_time_series(df_filtered, y_column='N2O', title='Filtered N2O Concentration Over Time', mode='markers')
```

![Filtered N2O](/assets/images/python/5/filtered_N2O.png)


This looks much better! The noise is gone, and a clear, meaningful pattern has emerged.

### 2.4 Understanding the Data Pattern
The filtered data shows a repeating pattern which is the signature of the static chamber method:
Baseline (Ambient Air): The long, relatively flat periods show the baseline N₂O concentration in the ambient air.
Concentration Increase (Chamber Closed): The sections where the concentration rises steadily and linearly are the actual measurements. This occurs when the chamber is placed over the soil, trapping the gases being emitted. The rate of this increase is what we will use to calculate the flux.
Sudden Drop (Chamber Opened): The sharp vertical drops occur when a measurement is finished, and the chamber is lifted from the ground, exposing the sensor to ambient air again.
Leveling Off: If a chamber is left on the ground for too long, the gas concentration inside can build up, altering the pressure gradient between the soil and the chamber air. This can cause the rate of increase to slow down and "level off." For this reason, it's crucial to use only the initial, linear part of the increase for our flux calculation.

## 3. Calculating flux for a single measurement
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

First, let's have a look on the fomula of flux calculation.

​
$$
\text{Flux Rate (molar)} = \frac{\frac{\Delta C}{t} \cdot V \cdot p}{R \cdot (T_{c} + 273.15) \cdot A}
$$
​

Where:

ΔC/t: The rate of change of the gas concentration in ppm/s (this will be the slope from our regression).

V: The total volume of the chamber headspace (m³).

p: The air pressure in Pascals (Pa) during measurement.

R: The ideal gas constant (8.314 J K⁻¹ mol⁻¹).

T_c: The air temperature in Celsius (°C).

A: The surface area covered by the chamber (m²).

To understand this fomula, we need to figure out the meaning of 'flux'. In the context of climate change, greenhouse gas flux specifically refers to the exchange of greenhouse gases (GHGs) like carbon dioxide (CO₂), methane (CH₄), and nitrous oxide (N₂O) between different parts of the Earth system (https://climate.sustainability-directory.com/term/greenhouse-gas-fluxes/#:~:text=In%20the%20context%20of%20climate,parts%20of%20the%20Earth%20system). Under the context of this analysis, 'flux' means gases exchange between soil and our measurement chamber. You might ask, "Doesn't the rate of concentration change, ΔC/t (in ppb/s), already represent this flux?" Actually, ΔC/t is the raw evidence of a flux, but it is not a standardized, comparable measurement. It only describes what's happening inside our specific chamber, under the specific conditions of that one measurement. We are not able to compare flux by simply comparing change rate of gas concentration. Under different temperature and pressure, gas molar density vary, the amount of gas molecular can be different even the gas volume is the same. Therefore, we need to utilize Gas Law (PV = nRT) to calculate the amount of molecular. Besides, a chamber covering a large area of soil will naturally capture more gas than one covering a small area. To make the measurement independent of our chamber's specifications, we must divide by the soil Area (A) it covers. By applying the full formula, We convert our raw observation (ΔC/t) into a robust, standardized unit: micromoles per square meter per second (µmol m⁻² s⁻¹).

To better understand the above fomula, it can be arranged into the following:


$$
\text{Flux Rate (molar)} = ( \frac{\Delta C}{t} \right) \cdot ( \frac{p \cdot V} {R \cdot (T_C + 273.15)} \right) \cdot ( \frac{1}{A} \right)
$$


Now, it is clear that the fomula only contains three components: **Flux** = **slope** * **gas_mole** * **A⁻¹**

Okay, lets create a function of flux calculation based on the fomula for later use.



<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">

{% capture exercise %}
### Exercise

The function `calculate_flux` is provided below but is not complete. It is your task to finish the function based on the formula.

```python
# Define key physical constants
R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)

def calculate_flux(slope_ppb_s, temp_k, pressure_pa, volume, area):
    """
    Calculates N2O flux.

    Parameters:
    - slope_ppb_s (float): Rate of change in ppb/s.
    - temperature (float): Temperature, assumed to be in Celsius or Kelvin.
    - pressure (float): Pressure, assumed to be in Pascals (Pa) or hectopascals (hPa).
    - volume (float): Chamber volume, assumed to be in cubic meters (m³) or Liters (L).
    - area (float): Chamber area, assumed to be in square meters (m²) or square cm (cm²).
    """
    # Convert slope from ppb/s to ppm/s for the formula
    ppm_per_second = ...
    
    # Calculate molar density of air (n/V = P/RT) in mol/m³
    gas_model = ...
    
    # Calculate the flux in µmol m⁻² s⁻¹
    # The 1e6 converts from mol to µmol
    flux = ...
    return flux
```

<details markdown="1">
<summary>Solution!</summary>
Here is the completed function:

```python

# Define key physical constants
R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)

def calculate_flux(slope_ppb_s, temp_k, pressure_pa, volume, area):
    """
    Calculates N2O flux.

    Parameters:
    - slope_ppb_s (float): Rate of change in ppb/s.
    - temperature (float): Temperature, assumed to be in Celsius or Kelvin.
    - pressure (float): Pressure, assumed to be in Pascals (Pa) or hectopascals (hPa).
    - volume (float): Chamber volume, assumed to be in cubic meters (m³) or Liters (L).
    - area (float): Chamber area, assumed to be in square meters (m²) or square cm (cm²).
    """
    # Convert slope from ppb/s to ppm/s for the formula
    ppm_per_second = slope_ppb_s / 1000.0
    
    # Calculate molar density of air (n/V = P/RT) in mol/m³
    gas_mole = (pressure_pa * volume)/ (R * temp_k)
    
    # Calculate the flux in µmol m⁻² s⁻¹
    # The 1e6 converts from mol to µmol
    flux = ppm_per_second * gas_mole / area * 1e6
    return flux

```
</details>

{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

### 3.2 Isolating and Visualizing the Measurement Data
Let's use an example time period of measurement: 2025-08-15 12:04:00 to 2025-08-15 12:10:00. We'll slice our df_filtered DataFrame to get only the data within this window and then plot it to get a closer look.

```python
# Define the start and end times for our measurement window
start_time = '2025-08-15 12:04:00'
end_time = '2025-08-15 12:09:30'

# Select the data for this specific time window
measurement_data = df_filtered[(df_filtered.index >= start_time) & (df_filtered.index < end_time)]

# Use our plotting function to visualize this specific period
plot_time_series(
    measurement_data, 
    y_column='N2O', 
    title=f'N2O Concentration for Plot {plot_metadata["plot_id"]}',
    mode='markers'
)
```
As you can see from the plot, the data in our 5-minute window 2025-08-15 12:04:00 - 2025-08-15 12:09:30 contains more than just the measurement itself. 

We can identify three distinct phases:

Pre-measurement Baseline: A flat period at the beginning. This is when the sensor was measuring ambient air before the chamber was placed on the collar.
The Measurement (Linear Increase): This is the part we want. The chamber is sealed, and N₂O from the soil is accumulating, causing a steady, linear increase in concentration.
Post-measurement Drop: The sharp, sudden drop at the end. This occurred when the chamber was lifted, and the sensor was exposed to ambient air again.

Our flux calculation relies on the slope (ΔC/t) from a linear regression. If we include the flat baseline or the sharp drop in our regression, the line of best fit will not represent the true rate of accumulation, leading to a highly inaccurate flux calculation.

Therefore, To get an accurate flux, visual inspection is necessary to include only the linear increase phase. Zoom in on the interactive Plotly graph. We can see that the clean, linear increase happens approximately between 12:05:30 and 12:09:00.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">

{% capture exercise %}
### Exercise
Try to slice the dataframe based on your refined time window, and plot it to see our refined result.

<details markdown="1">
<summary>Solution!</summary>
Here is the completed function:

```python
# Define the refined, visually inspected time window
start_mea = '2025-08-15 12:05:30'
end_mea = '2025-08-15 12:09:00'

# Create a new DataFrame with data only from this refined window
# We use .copy() to create a completely new object for the regression
measurement_data = df_filtered[df_filtered.index > start_mea & df_filtered.index < end_mea].copy()

# Visualize the refined data to confirm our selection
plot_time_series(
    regression_data, 
    y_column='N2O', 
    title=f'Refined Regression Window for Plot {plot_metadata["plot_id"]}',
    mode='markers'
)
```
</details>

{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>
Great! This plot shows the clear, linear increase in N₂O concentration after the chamber was placed on the collar. This is the exact data we need for our regression.

### 3.3 Linear Regression to derive the rate of gas concentration change
Now, as we talked before, we will fit a linear line to these data points. The slope of that line is the dC/dt (rate of change) that we need for our flux formula. As we expect the unit of our regression slope to be ppb/s our x-axis needs to be seconds elapsed (it means the seconds passed compared to the start of the measurement) instead of a timestamp. So, our first step is to create a new column, elapsed_seconds.

```python
from scipy import stats
measurement_data = measurement_data.copy()
# Create an 'elapsed_seconds' column for the regression
# First, we get the start time of the measurement
start_timestamp = measurement_data.index.min()
# Then we get the time difference for each time point and the start of measurement, and use function total_seconds convert this time difference into seconds
measurement_data['elapsed_seconds'] = (measurement_data.index - start_timestamp).total_seconds()
```

Then, we are going to actually fit the regression using scipy library. R2 represents the strength of the relationship that we detected. In here, we are going to use r2 = 0.7 as a threshold. If R2 of a regression is lower than 0.7, the change of gas concentration as time is not significant enough to be recognize as a flux (no flux is detected from the data), otherwise a gas flux can be deceted from the data.

```python
# Perform the linear regression using SciPy
slope, intercept, r_value, p_value, std_err = stats.linregress(
    x=measurement_data['elapsed_seconds'],
    y=measurement_data['N2O']
)

# The R-squared value tells us how well the line fits the data (a value > 0.7 is good!)
r_squared = r_value**2

print(f"--- Regression Results ---")
print(f"Slope (dC/dt): {slope:.4f} ppb/s")
print(f"R-squared: {r_squared:.4f}")
```
### 3.4 Visualizing the Fit and Final Calculation
It's always good practice to visualize the regression line against the data to confirm the fit is good.

```python
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
```
Good! If the regression is well fitted into our data, we are able to calculate the flux now! Before we call the calculate_flux function, there are still some steps to go. 
First, we need to get the average chamber air temperature and air pressure during the measurement and convert them into desired unit respectively (K for air temperature and Pa for air pressure). The unit conversion is very important, as when we wrote the calculate_flux function, we assumed units of our inputs. The mismatch of units will introduce systematic errors and leading to inaccuracy. 

```python
# try to get the average air temperature and air temperature value, don't forget the unit conversion.
avg_temp_c = 
avg_pressure_pa = 
```
<details markdown="1">
<summary>Solution!</summary>
```python
# Get the average temperature and pressure during the measurement
avg_temp_c = measurement_data['T_air'].mean() + 273.15 # convert from °C to K
avg_pressure_pa = measurement_data['P_air'].mean() * 100 # Assuming pressure is in hPa, convert to Pa
```
</details>
Then, we still need the total volume of the chamber headspace (m³) and the surface area covered by the chamber (m²). As they are independent of time and the same for all plots, we can simply define them as constants. 

```python
# --- Finally, Calculate the Flux! ---
VOLUME = 0.126 # 
AREA = 0.13 # the radias of the collar ring is 0.2m, so the area is 0.2*0.2*PI
```
Finally, call our calculate_flux function and we can get the result!

```python
# Now try to call the function using all the inputs we have and print out to check the result.
flux_N2O = ...

print(...)
```
<details markdown="1">
<summary>Solution!</summary>
```python
# Now we have all the pieces! Let's call our function.
flux_N2O = calculate_flux(
    slope_ppb_s=slope,
    temp_k=avg_temp_k,
    pressure_pa=avg_pressure_pa,
    volume=VOLUME,
    area=AREA

)

print(f"\n--- Final Flux Calculation ---")
print(f"Average Temperature: {avg_temp_c:.2f} °C")
print(f"Average Pressure: {avg_pressure_pa:.2f} Pa")
print(f"Calculated N₂O Flux: {flux_N2O:.5f} µmol m⁻² s⁻¹")
```
</details>

Brilliant! Now you successfuly turn the raw gas concentration data into gas fulx!

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">

{% capture exercise %}
### Challenge
Our current calculate_flux function works well, but it has a hidden weakness. It assumes the units of the inputs are correct. For example, it blindly assumes the temp_k argument is already in Kelvin. What if a user accidentally passes in a temperature in Celsius? The function would run without an error but produce a wildly incorrect result.
Code that relies on such hidden assumptions is sometimes called "hard-coded." A much better practice is to write more flexible code that can handle different situations or at least warn the user when something is wrong.

**Task**: upgrade the calculate_flux function to be more robust. It should:
1. Add Unit Checks: Check the input values to make a reasonable guess about their units.
2. Perform Automatic Unit Conversion: If it detects a value in a common but incorrect unit (like Celsius for temperature), it should automatically convert it to the required unit (Kelvin).
3. Raise Errors: If a value is completely outside a plausible range, it should stop and raise an error with a helpful message.

**Tip**: You can determine units by checking the physical range of a variable. For example, for a terrestrial field measurement, if a temperature value is between -50 and 50, it's almost certainly Celsius. If it's between 223 and 323, it's likely already in Kelvin.

<details markdown="1">
<summary>Solution!</summary>


```python

# Define key physical constants
R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)

def calculate_flux(slope_ppb_s, temperature, pressure, volume, area):
    """
    Calculates N2O flux with extensive unit checks and auto-conversion for all inputs.

    Parameters:
    - slope_ppb_s (float): Rate of change in ppb/s.
    - temperature (float): Temperature, assumed to be in Celsius or Kelvin.
    - pressure (float): Pressure, assumed to be in Pascals (Pa) or hectopascals (hPa).
    - volume (float): Chamber volume, assumed to be in cubic meters (m³) or Liters (L).
    - area (float): Chamber area, assumed to be in square meters (m²) or square cm (cm²).
    """
    # --- 1. Input Validation and Unit Conversion ---
    
    # Check Temperature (Celsius vs. Kelvin)
    if -50 <= temperature <= 50:
        print(f"Note: Temperature ({temperature}) detected as Celsius. Converting to Kelvin.")
        temp_k = temperature + 273.15
    elif 223 <= temperature <= 323:
        temp_k = temperature
    else:
        raise ValueError(f"Temperature value ({temperature}) is outside a plausible range.")

    # Check Pressure (Pascals vs. hPa)
    if 800 <= pressure <= 1100:
        print(f"Note: Pressure ({pressure}) detected as hPa/mbar. Converting to Pascals.")
        pressure_pa = pressure * 100
    elif 80000 <= pressure <= 110000:
        pressure_pa = pressure
    else:
        raise ValueError(f"Pressure value ({pressure}) is outside a plausible range.")
        
    # Check Volume (m³ vs. Liters)
    if 10000 <= volume <= 2000000
        print(f"Note: Volume ({volume}) detected as cm³. Converting to m³.")
        volume_m3 = volume / 1e6
    elif 10 <= volume <= 2000: # Plausible range for Liters
        print(f"Note: Volume ({volume}) detected as Liters. Converting to m³.")
        volume_m3 = volume / 1000.0
    elif 0.01 <= volume <= 2: # Plausible range for m³
        volume_m3 = volume
    else:
        raise ValueError(f"Volume value ({volume}) is outside a plausible range for m³ or Liters.")

    # Check Area (m² vs. cm²)
    if 100 <= area <= 20000: # Plausible range for cm²
        print(f"Note: Area ({area}) detected as cm². Converting to m².")
        area_m2 = area / 10000.0
    elif 1 <= area <= 200:  # Plausible range for dm²
        area_m2 = area / 100.0
        print(f"Note: Area ({area}) detected as dm². Converting to m².")
    elif 0.01 <= area <= 2: # Plausible range for m²
        area_m2 = area
    else:
        raise ValueError(f"Area value ({area}) is outside a plausible range for m², dm² or cm².")

    # --- 2. Core Calculation ---
    v_over_a = volume_m3 / area_m2
    ppm_per_second = slope_ppb_s / 1000.0
    molar_density = pressure_pa / (R * temp_k)
    flux = ppm_per_second * molar_density * v_over_a * 1e6
    
    return flux
```
**Info**: Python ([raise keywords](https://www.geeksforgeeks.org/python/python-raise-keyword/)) is used to raise exceptions or errors. The raise keyword raises an error and stops the control flow of the program. It is used to bring up the current exception in an exception handler (an exception handler indicates the error type) so that it can be handled further up the call stack. 

The basic way to raise an exception is 
```python
raise Exception ('...') # In here, Exception is an exception handler (it is actually a function), which indicate a general exception. It takes a string used to reminder                            # users what errors happen in here and the potential reasons. 
```

In the 'solution', we used the handler ValueError to indicate the input value is outside a plausible range, and pass a string showing to users to further explain the error.
</details>

{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>



## 4. Automating gas flux calculation
### 4.1 Store and structure measurement info 
 
The first and crucial step of automation is to store the key information (metadata) for each measurement in a structured way that a program can loop through. For this, we will use a Python dictionary. The dictionary keys will be our data "columns" (e.g., 'plot_id', 'land_use'), and the values will be lists containing the data for each plot.
Now, there is an issue: we take multiple measurements at the same plot, perhaps on different days or at different times. How can we store this information efficiently?
We can store the multiple start and end times for a single plot as a single string, with each timestamp separated by a semicolon (;). Of course, the order of the multiple starttime and endtime for a plot should match. By doing this, we can keep the metadata table concise and still tell our program to perform multiple calculations for that plot.



```python
# --- Create Metadata ---
measurement_info = {
    'plot_id': [1, 2],
    'land_use': ['forest', 'forest'],
    'start_time': ['2025-08-15 12:06:00; 2025-08-15 12:14:00', '2025-08-15 12:13:00'],
    'end_time': ['2025-08-15 12:09:00; 2025-08-15 12:17:00', '2025-08-15 12:18:30'],
}
metadata_df = pd.DataFrame(measurement_info)
```
### 4.2 Automation Calculation
Now we can build a for loop that iterates through each row (Each row contains infomation for a single plot) of our metadata_df. In here, we are going to use ['iterrow()'](https://www.geeksforgeeks.org/pandas/pandas-dataframe-iterrows/) to iterate through metadata_df. 'iterrow()' is a method of data frame object, it generates an iterator object of the DataFrame, allowing us to iterate each row in the DataFrame. Each iteration produces an index object and a row object (a Pandas Series object). Inside the loop, we will split start_time and end_time string for each plot using 'split()' method and build a inner loop to iterate all measurements for the plot.
```python
results = [] # Create a empty list we can use to store all calculated fluxes.
for index, row in metadata_df.iterrows():
    start_times = row['start_time'].strip().split(';')  # Handle potential multiple times
    end_times = row['end_time'].strip().split(';')  # Handle potential multiple times

    for start_time, end_time in zip(start_times, end_times):
        start_time = pd.to_datetime(start_time.strip())
        end_time = pd.to_datetime(end_time.strip())
        measurement_date = f'{start_time.year}-{start_time.month:02d}-{start_time.day:02d}'
```
Within the inner loop, we will perform the exact same steps we did manually in the last section. However, there is one key difference in the visual inspection step. In the manual section, we looked at the plot and then assigned our refined start and end times into variables. To keep the program continuing without needing to stop and edit the script each time, we will use the built-in input() function. This will pause the script, show us a plot, and allow us to enter our refined time window directly into the terminal before the program continues.
```python
        ## step 1: Visual inspection ##
        # Select the data for this specific time window
        measurement_data = df_filtered[(df_filtered.index >= start_time) & (df_filtered.index < end_time)]
        # Plot the raw data for visual inspection
        plot_time_series(measurement_data, y_column='N2O', title=f'N2O Concentration Over Time}', mode='markers')
        # Mannually selcect the start and end time for regression
        start_mea = input("Enter the start time for regression (YYYY-MM-DD HH:MM:SS): ").strip()
        end_mea = input("Enter the end time for regression (YYYY-MM-DD HH:MM:SS): ").strip()
        # Use the original start and end time if no input is given
        if not start_mea:
            start_mea = start_time
        if not end_mea:
            end_mea = end_time
        start_time = pd.to_datetime(start_mea)
        end_time = pd.to_datetime(end_mea)
        measurement_data = measurement_data[(measurement_data.index >= start_time) & (measurement_data.index <= end_time)]
        ## step 2: Linear regression ## 
        # Ensure there is enough data to perform a regression
        if len(measurement_data) < 10:
            print(f"Skipping plot {row['plot_id']} on {measurement_date} due to insufficient data.")
            continue

        # Create an 'elapsed_seconds' column for the regression
        measurement_data['elapsed_seconds'] = (measurement_data.index - start_time).total_seconds()

        # Perform linear regression: N2O concentration vs. time
        slope, intercept, r_value, p_value = stats.linregress(
            x=measurement_data['elapsed_seconds'],
            y=measurement_data['N2O']
        )

        # --- Quality Control (QC) ---
        # We only accept measurements with a good linear fit and a positive slope
        r_squared = r_value**2
        # plot the regression line
        fig, ax = plt.subplots(layout='constrained', figsize=(10, 5))
        ax.scatter(measurement_data['elapsed_seconds'], measurement_data['N2O'], label='N2O Concentration (ppb)')
        ax.plot(measurement_data['elapsed_seconds'], intercept + slope * measurement_data['elapsed_seconds'], 'r', label='Fitted line')
        ax.set_xlabel('Elapsed Time (s)')
        ax.set_ylabel('N2O Concentration (ppb)')
        ax.set_title(f'Linear Regression for Plot {row["plot_id"]} (R²={r_squared:.2f})')
        plt.legend()
        plt.show()

        if r_squared < 0.70 or p_value > 0.05:
            flux_umol_m2_s = 0  # Set flux to 0 if QC fails
            qc_pass = False
        else:
            qc_pass = True
            
            ## step 3: Flux Calculation Formula ##
            # This formula converts the rate of change in concentration (slope) to a flux rate.
            # It corrects for ambient pressure and temperature.
            
            temp_k = measurement_data['temperature'].mean() + 273.15  # Convert °C to Kelvin
            pressure_pa = measurement_data['pressure'].mean() * 100  # Convert hPa to Pascals

            flux_umol_m2_s = calculate_flux(slope, temp_k, pressure_pa, VOLUME, AREA)
```
At the end of the iteration, we need to save the results of each calculation. Only the flux value is not enough, we also need to save its metadata (e.g., plot_id, 'land_use'), which are essential for flux analysis and visualization we are going to do later.
```python
        # Store the results
        results.append({
            'plot_id': row['plot_id'],
            'land_use': row['land_use'],
            'measurement_date': measurement_date,
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
```
### 4.3 Flux comparison
```python
# --- Visualization ---
plt.figure(figsize=(10, 7))
sns.boxplot(data=flux_results_df, x='land_use', y='N2O_flux_umol_m2_s', palette='viridis')
sns.stripplot(data=flux_results_df, x='land_use', y='N2O_flux_umol_m2_s', color='black', size=8, jitter=True, alpha=0.7)

plt.title('N₂O Flux by Land Use Type', fontsize=16)
plt.xlabel('Land Use', fontsize=12)
plt.ylabel('N₂O Flux (µmol m⁻² s⁻¹)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
