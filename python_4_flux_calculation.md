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
with open("./BAI_StudyProject_LuentenerWald/raw_data/N2O/TG20-01072-2025-08-15T110000.data.txt") as f:
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

The GGA analyzer produces comma-separated files with a different structure: The first line includes instrument metadata (version, serial number, etc.), The second are column headers, Lines 3+ hold measurement data.

Here's an example of the first few lines:

```
VC:2f90039 BD:Jan 16 2014 SN:
                     Time,      [CH4]_ppm,   [CH4]_ppm_sd,      [H2O]_ppm, ...
  08/15/2025 11:00:03.747,   2.080375e+00,   0.000000e+00,   1.103072e+03, ...
```

However, GGA files contain extra non-data content at the end (such as digital signatures or log messages). We need to filter these out. Let's build our loader step by step.

### Read the CSV File

First, we read the file with `pd.read_csv()`, skipping the first metadata line:

```python
df_gga = pd.read_csv(
    "./BAI_StudyProject_LuentenerWald/raw_data/GGA/gga_2025-08-15_f0000.txt",
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
# As regular expression can only be applied on string, we need to make sure timestamp is in string format
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
df_gga = load_gga_data("./BAI_StudyProject_LuentenerWald/raw_data/GGA/gga_2025-08-15_f0000.txt")
print(f"Loaded {len(df_gga):,} rows")
print(f"Time range: {df_gga.index.min()} to {df_gga.index.max()}")
df_gga.tail()
```



### 1.3 Loading Multiple Files
Now that we have loader functions, we can easily handle data from multiple field trips. Instead of copying code, we can simply call our function in a loop.
First, we create a list of all the file paths we want to load. Then, we can loop through this list, call our function for each path, and store the resulting DataFrames in a new list.

```python
# First, let's list all the files we want to load.
# Make sure the file paths are complete and correct.
base_path = "./BAI_StudyProject_LuentenerWald/raw_data/"
n2o_files = [
    'N2O/TG20-01072-2025-08-15T110000.data.txt',
    'N2O/TG20-01072-2025-08-26T093000.data.txt'
]

gga_files = [
    'GGA/gga_2025-08-15_f0000.txt',
    'GGA/gga_2025-08-06_f0000.txt',
    'GGA/gga_2025-08-26_f0000.txt'
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
base_path = "./BAI_StudyProject_LuentenerWald/raw_data/Ta/"

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

  
## 2. Visualizing and Cleaning the Data

Now that we have a single, merged DataFrame, our next step is to inspect the data quality. Raw sensor data from the field is almost never perfect. Visualizing it is the best way to diagnose issues like noise, drift, or outliers before we attempt any calculations. 

In this section, we will:
1. Create a reusable plotting function with Plotly
2. Visualize the raw gas data to identify problems
3. Apply a quantile filter to remove outliers
4. Understand the patterns in our cleaned data


### 2.1 Creating a Reusable Plotting Function with Plotly

For visualization, we'll use **Plotly**, a powerful library for creating interactive plots. Unlike static plots from Matplotlib, Plotly allows you to zoom, pan, and hover over data points—perfect for inspecting time-series data. Just as we did with data loading, we'll be plotting our time-series data multiple times. To make this efficient and keep our plots looking consistent, let's create a dedicated function.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise
The plotting function is partly provided below. Complete the function by filling in the `add_trace()` and `update_layout()` calls!

**Hints:**
- Use `go.Scatter()` for the trace with `x=`, `y=`, `mode=`, and `name=` parameters
- The layout should include `title`, `xaxis_title`, `yaxis_title`, and `template`

```python
import plotly.graph_objects as go
import plotly.io as pio

# This setting forces Plotly to open plots in your default web browser
pio.renderers.default = "browser"

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

    fig.add_trace(...)  # TODO: Add a Scatter trace

    fig.update_layout(
        ...  # TODO: Set title, axis labels, and template
    )
    
    fig.show()
```

<details markdown="1">
<summary>Click here for the solution!</summary>
    
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
    
    # Work on a copy to avoid changing the user's original DataFrame
    df_plot = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_plot.index):
        print("Note: The DataFrame index is not a DatetimeIndex.")
        if 'Timestamp' in df_plot.columns:
            print("--> Found a 'Timestamp' column. Setting it as the index.")
            df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'])
            df_plot = df_plot.set_index('Timestamp')
        else:
            raise TypeError(
                "The DataFrame index is not a DatetimeIndex and a 'Timestamp' "
                "column was not found. Please set a DatetimeIndex before plotting."
            )
            
    # --- Plotting ---
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
        yaxis_title=y_column, 
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

Now, let's use our new function to look at the raw N₂O data. The interactive plot allows you to zoom and pan to inspect noisy areas.

```python
# 1. Ensure Timestamp is the index and sorted
# Only set index if 'Timestamp' is currently a column
if 'Timestamp' in df_merged.columns:
    df_merged = df_merged.set_index('Timestamp')

df_merged = df_merged.sort_index()

# 2. Select where N2O has valid values (Drop NaNs)
df_N2O = df_merged[['N2O_ppb']].dropna() 

# 3. Plot
plot_time_series(
    df_N2O, 
    y_column='N2O_ppb', 
    title='N₂O Concentration',
    mode='markers'
)
```

![raw data plotting](/assets/images/python/5/raw_data_plot.png)

**What do we see?**

The raw data is very noisy! There are several problems:
- **Negative values**: Physically impossible for gas concentrations
- **Extreme spikes**: Values far outside the expected range
- **Sensor noise**: Random fluctuations due to electrical interference

These artifacts are common in field measurements and must be removed before we can calculate meaningful fluxes.


### 2.3 Filtering with a Quantile Filter

To remove outliers, we'll use a **quantile filter**, it can help us see the real data patterns (signal) by removing all outliers/noise. This method calculates percentiles of the data and keeps only values within a specified range.

### Why Quantile Filtering?

Quantile filtering is **robust to outliers**. Unlike methods based on mean and standard deviation, extreme values have very little influence on percentile calculations. This makes it ideal for sensor data with occasional spikes.

The approach:
1. Calculate the 3th percentile ($P_{3}$) and 97th percentile ($P_{97}$)
2. Keep only data points where: $P_{3} \leq x \leq P_{97}$
3. Discard everything else

> **Info: What are Quantiles?**
> 
> A **quantile** divides your data into equal-sized groups. Common examples:
> - **Median** (50th percentile): Half the data is below, half is above
> - **Quartiles**: Divide data into 4 parts (25th, 50th, 75th percentiles)
> - **Percentiles**: Divide data into 100 parts
>
> The 10th percentile means "10% of values are below this point."
>
> **Resources:**
> - [Pandas quantile() documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html)
> - [Understanding Percentiles (Khan Academy)](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/percentile-rankings/v/calculating-percentile)

### Applying the Filter

```python
# Calculate the 3rd and 97th percentiles
p_3 = df_N2O['N2O_ppb'].quantile(0.03)
p_97 = df_N2O['N2O_ppb'].quantile(0.97)

print(f"3rd percentile:  {p_3:.2f} ppb")
print(f"97th percentile: {p_97:.2f} ppb")
print(f"Keeping data in range [{p_3:.2f}, {p_97:.2f}]")

# Apply the filter
df_filtered = df_N2O[
    (df_N2O['N2O_ppb'] >= p_3) & 
    (df_N2O['N2O_ppb'] <= p_97)
].copy()

# Calculate statistics based on the N2O dataframe, not the merged one
n_raw = len(df_N2O)
n_clean = len(df_filtered)
n_removed = n_raw - n_clean

print(f"\nValid rows before filtering: {n_raw:,}")
print(f"Rows after filtering:        {n_clean:,}")
print(f"Outliers removed:            {n_removed:,} ({(n_removed/n_raw)*100:.1f}%)")
```

Now let's visualize the filtered data:

```python
# Plot filtered data using markers to see individual points
plot_time_series(
    df_filtered, 
    y_column='N2O_ppb', 
    title='Filtered N₂O Concentration Over Time', 
    mode='markers'
)
```

![Filtered N2O](/assets/images/python/5/filtered_N2O.png)

This looks much better! The noise is gone, now please pan and zoom in to check the N2O data measured on 15th and 26th of Aug. You can see a clear, meaningful pattern.

![Filtered_N2O_0815](/assets/images/python/5/Filtered_N2O_0815.png)

![Filtered_N2O_0826](/assets/images/python/5/Filtered_N2O_0826.png)

### 2.4 Understanding the Data Pattern

The filtered data reveals a repeating pattern characteristic of the **static chamber method**. Let's break down what we're seeing:

| Phase | What's Happening | What You See in the Plot |
|-------|------------------|--------------------------|
| **1. Baseline** | Chamber is open, sensor measures ambient air | Long, flat periods at ~background concentration |
| **2. Accumulation** | Chamber closed over soil, gases accumulate | Steady, linear increase in concentration |
| **3. Release** | Chamber lifted, gases escape | Sharp vertical drop back to baseline |
| **4. Leveling off** | (If chamber left too long) Soil-air gradient decreases | Rate of increase slows, curve flattens |

### The Critical Insight: Linear Increase

The **rate of concentration increase** during the accumulation phase is what we use to calculate flux. Mathematically:

$$\text{Flux} \propto \frac{dC}{dt}$$

Where:
- $C$ = gas concentration inside the chamber
- $t$ = time
- $\frac{dC}{dt}$ = rate of change (slope of the linear portion)

> **Important: Use Only the Linear Portion**
> 
> If a chamber is left on the ground too long, gas buildup inside the chamber reduces the concentration gradient between soil and chamber air. This causes the accumulation rate to slow down ("leveling off").
> 
> For accurate flux calculations, we must identify and use **only the initial, linear part** of each accumulation period. We'll learn how to do this in the next section.


The same visualization and filtering workflow applies to the GGA data (CH₄ and CO₂). But does the GGA data actually need filtering? Let's find out!

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise: Inspect and Clean the GGA Data

**Part 1: Visualize the raw data**

First, plot the raw CH₄ and CO₂ data to see if they contain noise or outliers.

```python
# Plot raw CH4 data
plot_time_series(
    df_merged[df_merged['CH4_ppm'].notna()], 
    y_column='CH4_ppm', 
    title='Raw CH₄ Concentration Over Time', 
    mode='markers'
)

# Plot raw CO2 data
plot_time_series(
    df_merged[df_merged['CO2_ppm'].notna()], 
    y_column='CO2_ppm', 
    title='Raw CO₂ Concentration Over Time', 
    mode='markers'
)
```

**Questions to consider:**
1. Do you see any obvious outliers or impossible values (like negative concentrations)?
2. Are there extreme spikes that look like sensor errors?
3. Use the zoom and pan features to inspect different time periods.

**Part 2: Decide on filtering**

Based on your visual inspection:
- **If the data looks clean:** No filtering needed! Move on to the next section.
- **If the data contains outliers:** Apply a quantile filter like we did for N₂O.

**Part 3: Apply filtering (if needed)**

If you determined that filtering is necessary, apply the quantile filter to CH₄ and CO₂.

**Hints:**
- Use `.quantile(0.03)` and `.quantile(0.97)` to get the 10th and 90th percentiles

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# 1. Create clean copies for CH4 and CO2 (removing NaNs first simplifies everything)
df_CH4 = df_merged[['CH4_ppm']].dropna().copy()
df_CO2 = df_merged[['CO2_ppm']].dropna().copy()

# --- Filter CH4 data (3rd - 97th percentile) ---
ch4_p03 = df_CH4['CH4_ppm'].quantile(0.03)
ch4_p97 = df_CH4['CH4_ppm'].quantile(0.97)

print(f"CH4 filter range: [{ch4_p03:.3f}, {ch4_p97:.3f}] ppm")

# Apply filter to the specific dataframe
df_CH4_clean = df_CH4[
    (df_CH4['CH4_ppm'] >= ch4_p03) & 
    (df_CH4['CH4_ppm'] <= ch4_p97)
].copy()

# --- Filter CO2 data (3rd - 97th percentile) ---
co2_p03 = df_CO2['CO2_ppm'].quantile(0.03)
co2_p97 = df_CO2['CO2_ppm'].quantile(0.97)

print(f"CO2 filter range: [{co2_p03:.1f}, {co2_p97:.1f}] ppm")

# Apply filter to the specific dataframe
df_CO2_clean = df_CO2[
    (df_CO2['CO2_ppm'] >= co2_p03) & 
    (df_CO2['CO2_ppm'] <= co2_p97)
].copy()
```

```python
plot_time_series(
    df_CH4_clean, 
    y_column='CH4_ppm', 
    title='Filtered CH₄ Concentration (3rd-97th Percentile)', 
    mode='markers' # Markers are safer for high-frequency data
)

# Visualize the cleaned CO2 data
plot_time_series(
    df_CO2_clean, 
    y_column='CO2_ppm', 
    title='Filtered CO₂ Concentration (3rd-97th Percentile)', 
    mode='markers'
)
```

**Note:** Depending on your dataset, you may find that the GGA data is already quite clean and filtering removes very few points. This is because the Los Gatos GGA analyzer tends to produce more stable measurements than some other sensors. Always visualize first before deciding to filter!

</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

## 3. Calculating Flux for a Single Measurement

After loading and filtering our raw data and getting an overview of the patterns, it's time to calculate the fluxes. Excited?

In this section, we will focus on a **single measurement period** to understand the process in detail. We'll break it down into these key steps:

1. Review the flux calculation formula to understand what components we need
2. Isolate the data for a specific time window and visualize it
3. Perform a linear regression to get the rate of concentration change
4. Combine all the pieces to calculate the final flux

---

### 3.1 The Flux Calculation Formula

Before we start coding, let's understand the physics behind flux calculation.

#### What is "Flux"?

In the context of greenhouse gas research, **flux** refers to the exchange of gases between different parts of the Earth system—in our case, between the **soil** and the **atmosphere** inside our measurement chamber.

> **Learn More:** [Greenhouse Gas Fluxes - Sustainability Directory](https://climate.sustainability-directory.com/term/greenhouse-gas-fluxes/)

You might ask: *"Doesn't the rate of concentration change (ΔC/t) already represent the flux?"*

Not quite! The raw concentration change rate (in ppb/s) is evidence of a flux, but it's **not standardized**. It only describes what's happening inside our specific chamber under specific conditions. We cannot compare measurements directly because:

- **Gas density varies with temperature and pressure** — The same volume can contain different amounts of gas molecules
- **Chamber size matters** — A larger chamber captures more gas, giving a higher concentration change rate

To make measurements comparable, we need to convert our raw observation into a standardized unit: **µmol m⁻² s⁻¹** (micromoles per square meter per second).

#### The Formula

The flux calculation formula is:

$$
\text{Flux} = \frac{\Delta C / \Delta t \cdot V \cdot P}{R \cdot T \cdot A}
$$

Where:

| Symbol | Description | Unit |
|--------|-------------|------|
| $\Delta C / \Delta t$ | Rate of concentration change (slope from regression) | ppm s⁻¹ |
| $V$ | Chamber headspace volume | m³ |
| $P$ | Air pressure | Pa |
| $R$ | Ideal gas constant | 8.314 J K⁻¹ mol⁻¹ |
| $T$ | Air temperature | K (Kelvin) |
| $A$ | Surface area covered by chamber | m² |

#### Understanding the Formula

To better understand the formula, we can rearrange it into three intuitive components:

$$
\text{Flux} = \underbrace{\frac{\Delta C}{\Delta t}}_{\text{slope}} \times \underbrace{\frac{P \cdot V}{R \cdot T}}_{\text{moles of gas}} \times \underbrace{\frac{1}{A}}_{\text{per unit area}}
$$

This shows that: **Flux = Slope × Gas Moles × Area⁻¹**

> **Info: The Ideal Gas Law**
> 
> The middle term comes from the Ideal Gas Law: $PV = nRT$
> 
> Rearranging for $n$ (number of moles):
> $$n = \frac{PV}{RT}$$
> 
> This converts our concentration measurement into an actual amount of gas molecules.
>
> **Resources:**
> - [Ideal Gas Law (Khan Academy)](https://www.khanacademy.org/science/physics/thermodynamics/temp-kinetic-theory-ideal-gas-law/a/what-is-the-ideal-gas-law)
> - [Gas Laws (Chemistry LibreTexts)](https://chem.libretexts.org/Bookshelves/General_Chemistry/Map%3A_Chemistry_-_The_Central_Science_(Brown_et_al.)/10%3A_Gases/10.04%3A_The_Ideal_Gas_Law)

#### Creating the Flux Calculation Function

Now let's implement this formula as a Python function.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise

The function `calculate_flux` is provided below but is incomplete. Fill in the missing parts based on the formula.

**Hints:**
- Convert slope from ppb/s to ppm/s by dividing by 1000
- Use the Ideal Gas Law to calculate moles: $n = \frac{PV}{RT}$
- Multiply by 10⁶ to convert from mol to µmol

```python
# Define the ideal gas constant
R = 8.314  # J K⁻¹ mol⁻¹

def calculate_flux(slope_ppb_s, temp_k, pressure_pa, volume_m3, area_m2):
    """
    Calculates gas flux from chamber measurements.

    Parameters:
    - slope_ppb_s (float): Rate of concentration change in ppb/s
    - temp_k (float): Temperature in Kelvin
    - pressure_pa (float): Pressure in Pascals
    - volume_m3 (float): Chamber volume in cubic meters
    - area_m2 (float): Chamber area in square meters
    
    Returns:
    - float: Flux in µmol m⁻² s⁻¹
    """
    # Convert slope from ppb/s to ppm/s
    slope_ppm_s = ...
    
    # Calculate moles of gas using Ideal Gas Law (n = PV/RT)
    gas_moles = ...
    
    # Calculate flux in µmol m⁻² s⁻¹
    # Multiply by 1e6 to convert mol to µmol
    flux = ...
    
    return flux
```

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# Define the ideal gas constant
R = 8.314  # J K⁻¹ mol⁻¹

def calculate_flux(slope_ppb_s, temp_k, pressure_pa, volume_m3, area_m2):
    """
    Calculates gas flux from chamber measurements.

    Parameters:
    - slope_ppb_s (float): Rate of concentration change in ppb/s
    - temp_k (float): Temperature in Kelvin
    - pressure_pa (float): Pressure in Pascals
    - volume_m3 (float): Chamber volume in cubic meters
    - area_m2 (float): Chamber area in square meters
    
    Returns:
    - float: Flux in µmol m⁻² s⁻¹
    """
    # Convert slope from ppb/s to ppm/s
    slope_ppm_s = slope_ppb_s / 1000.0
    
    # Calculate moles of gas using Ideal Gas Law (n = PV/RT)
    gas_moles = (pressure_pa * volume_m3) / (R * temp_k)
    
    # Calculate flux in µmol m⁻² s⁻¹
    # Multiply by 1e6 to convert mol to µmol
    flux = slope_ppm_s * gas_moles / area_m2 * 1e6
    
    return flux
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

---

### 3.2 Isolating and Visualizing the Measurement Data

Now let's apply our formula to real data. We'll use an example measurement period from our field campaign.

#### Step 1: Define the Time Window

Let's select a measurement window from our filtered data:

```python
# Define the start and end times for our measurement window
start_time = '2025-08-15 12:04:00'
end_time = '2025-08-15 12:09:30'

# Select the data for this specific time window
measurement_data = df_filtered[
    (df_filtered.index >= start_time) & 
    (df_filtered.index < end_time)
]

print(f"Selected {len(measurement_data)} data points")
print(f"Time range: {measurement_data.index.min()} to {measurement_data.index.max()}")
```

#### Step 2: Visualize the Raw Measurement Window

```python
# Plot the measurement window
plot_time_series(
    measurement_data, 
    y_column='N2O_ppb', 
    title='N₂O Concentration - Full Measurement Window',
    mode='markers'
)
```

#### Understanding the Pattern

Looking at the plot, we can identify **three distinct phases**:

| Phase | Description | What You See |
|-------|-------------|--------------|
| **1. Pre-measurement Baseline** | Sensor measuring ambient air before chamber placement | Flat period at the beginning |
| **2. Accumulation (Linear Increase)** | Chamber sealed, N₂O accumulating from soil | Steady, linear rise ✓ |
| **3. Post-measurement Drop** | Chamber lifted, sensor exposed to ambient air | Sharp, sudden drop |

> **Critical: Use Only the Linear Phase**
> 
> Our flux calculation relies on the **slope** from linear regression. If we include the flat baseline or the sharp drop, the regression line will not represent the true accumulation rate, leading to **inaccurate flux values**.
> 
> We must visually inspect the data and select **only the linear increase phase**.

#### Step 3: Refine the Time Window

Use the zoom and pan features on the interactive plot to identify the linear portion. In this example, the clean linear increase occurs approximately between **12:05:30** and **12:09:00**.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise

Slice the DataFrame to include only the linear accumulation phase, then plot it to verify your selection.

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# Define the refined time window (linear portion only)
start_linear = '2025-08-15 12:05:30'
end_linear = '2025-08-15 12:09:00'

# Create a new DataFrame with only the linear phase
# Use .copy() to avoid SettingWithCopyWarning
regression_data = df_filtered[
    (df_filtered.index > start_linear) & 
    (df_filtered.index < end_linear)
].copy()

print(f"Selected {len(regression_data)} points for regression")

# Visualize to confirm our selection
plot_time_series(
    regression_data, 
    y_column='N2O_ppb', 
    title='Refined Regression Window (Linear Phase Only)',
    mode='markers'
)
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

Great! The plot should now show a clear, linear increase in N₂O concentration. This is exactly what we need for our regression.

---

### 3.3 Linear Regression to Derive the Rate of Change

Now we'll fit a linear regression line to our data. The **slope** of this line is the $\frac{\Delta C}{\Delta t}$ we need for our flux formula.

#### Step 1: Convert Timestamps to Elapsed Seconds

For regression, we need numeric x-values. We'll convert timestamps to "seconds elapsed since start of measurement":

```python
from scipy import stats

# Work on a copy to avoid modifying the original
regression_data = regression_data.copy()

# Get the start time
start_timestamp = regression_data.index.min()

# Calculate elapsed seconds for each data point
regression_data['elapsed_seconds'] = (
    regression_data.index - start_timestamp
).total_seconds()

# Check the result
print(regression_data[['elapsed_seconds', 'N2O_ppb']].head())
```

#### Step 2: Perform Linear Regression

We'll use SciPy's `linregress` function to fit a line:

```python
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    x=regression_data['elapsed_seconds'],
    y=regression_data['N2O_ppb']
)

# Calculate R-squared (coefficient of determination)
r_squared = r_value ** 2

print("--- Regression Results ---")
print(f"Slope (ΔC/Δt): {slope:.4f} ppb/s")
print(f"Intercept: {intercept:.2f} ppb")
print(f"R-squared: {r_squared:.4f}")
print(f"P-value: {p_value:.2e}")
```

> **Info: Interpreting R-squared**
> 
> The **R-squared** ($R^2$) value tells us how well the line fits the data:
> - $R^2 = 1.0$: Perfect fit
> - $R^2 > 0.9$: Excellent fit
> - $R^2 > 0.7$: Good fit (acceptable for flux calculation)
> - $R^2 < 0.7$: Poor fit (flux may not be reliable)
> 
> If $R^2 < 0.7$, the concentration change may not be significant enough to calculate a meaningful flux.
>
> **Resources:**
> - [SciPy linregress documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)
> - [Linear Regression (Khan Academy)](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/introduction-to-trend-lines/a/linear-regression-review)

---

### 3.4 Visualizing the Regression Fit

Before calculating the flux, let's visualize the regression line to confirm it fits well:

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add the raw data points
fig.add_trace(go.Scatter(
    x=regression_data['elapsed_seconds'], 
    y=regression_data['N2O_ppb'], 
    mode='markers', 
    name='Measured Data',
    marker=dict(size=8)
))

# Add the fitted regression line
fig.add_trace(go.Scatter(
    x=regression_data['elapsed_seconds'], 
    y=intercept + slope * regression_data['elapsed_seconds'],
    mode='lines', 
    name=f'Fitted Line (R²={r_squared:.3f})', 
    line=dict(color='red', width=2)
))

fig.update_layout(
    title=f'Linear Regression: Slope = {slope:.4f} ppb/s',
    xaxis_title='Elapsed Time (seconds)', 
    yaxis_title='N₂O Concentration (ppb)', 
    template='plotly_white',
    legend=dict(x=0.02, y=0.98)
)

fig.show()
```

If the red line closely follows the data points and $R^2 > 0.7$, we can proceed with confidence!

---

### 3.5 Final Flux Calculation

Now we have all the pieces. Let's put them together!

#### Step 1: Get Average Temperature and Pressure

We need the mean temperature and pressure during the measurement. **Unit conversion is critical!**

```python
# Get average temperature (convert °C to Kelvin)
avg_temp_c = regression_data['Ta_C'].mean()
avg_temp_k = avg_temp_c + 273.15

# Get average pressure (assuming data is in hPa, convert to Pa)
# Note: Check your data to confirm the original unit!
avg_pressure_hpa = regression_data['P_Pa'].mean()  # If already in Pa, skip conversion
avg_pressure_pa = avg_pressure_hpa  # Adjust if needed: * 100 for hPa to Pa

print(f"Average Temperature: {avg_temp_c:.2f} °C = {avg_temp_k:.2f} K")
print(f"Average Pressure: {avg_pressure_pa:.0f} Pa")
```

#### Step 2: Define Chamber Dimensions

The chamber volume and area are constants for our setup:

```python
# Chamber specifications (measure these for your specific equipment!)
CHAMBER_VOLUME = 0.0126  # m³ (example: 12.6 liters)
COLLAR_AREA = 0.1257     # m² (example: circle with radius 0.2 m → π × 0.2²)

print(f"Chamber Volume: {CHAMBER_VOLUME} m³")
print(f"Collar Area: {COLLAR_AREA} m²")
```

#### Step 3: Calculate the Flux

```python
# Calculate the flux!
flux_n2o = calculate_flux(
    slope_ppb_s=slope,
    temp_k=avg_temp_k,
    pressure_pa=avg_pressure_pa,
    volume_m3=CHAMBER_VOLUME,
    area_m2=COLLAR_AREA
)

print("\n" + "="*50)
print("        FINAL FLUX CALCULATION RESULT")
print("="*50)
print(f"  Slope:       {slope:.4f} ppb/s")
print(f"  Temperature: {avg_temp_k:.2f} K")
print(f"  Pressure:    {avg_pressure_pa:.0f} Pa")
print(f"  Volume:      {CHAMBER_VOLUME} m³")
print(f"  Area:        {COLLAR_AREA} m²")
print("-"*50)
print(f"  N₂O Flux:    {flux_n2o:.5f} µmol m⁻² s⁻¹")
print("="*50)
```

**Congratulations!** You've successfully converted raw gas concentration data into a standardized flux value!

---

### 3.6 Challenge: Making the Function More Robust

<div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 5px; border-left: 4px solid #ffc107;">
{% capture exercise %}
### Challenge Exercise

Our `calculate_flux` function works, but it has a hidden weakness: it assumes the user provides inputs in the correct units. What if someone accidentally passes temperature in Celsius instead of Kelvin? The function would run without error but produce **wildly incorrect results**.

**Your Task:** Upgrade the function to be more robust by:

1. **Detecting units** based on plausible value ranges
2. **Auto-converting** common unit mistakes
3. **Raising errors** for implausible values

**Tips for detecting units:**

| Variable | If value is in range... | It's probably... |
|----------|------------------------|------------------|
| Temperature | -50 to 60 | Celsius |
| Temperature | 220 to 330 | Kelvin |
| Pressure | 800 to 1100 | hPa (hectopascals) |
| Pressure | 80,000 to 110,000 | Pa (pascals) |
| Volume | 1 to 1000 | Liters |
| Volume | 0.001 to 1 | m³ |
| Area | 100 to 10,000 | cm² |
| Area | 0.01 to 1 | m² |

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
R = 8.314  # Ideal gas constant (J K⁻¹ mol⁻¹)

def calculate_flux_robust(slope_ppb_s, temperature, pressure, volume, area):
    """
    Calculates gas flux with automatic unit detection and conversion.

    Parameters:
    - slope_ppb_s (float): Rate of change in ppb/s
    - temperature (float): Temperature in Celsius or Kelvin (auto-detected)
    - pressure (float): Pressure in Pa or hPa (auto-detected)
    - volume (float): Chamber volume in m³ or Liters (auto-detected)
    - area (float): Chamber area in m² or cm² (auto-detected)
    
    Returns:
    - float: Flux in µmol m⁻² s⁻¹
    """
    
    # --- Temperature Check ---
    if -50 <= temperature <= 60:
        print(f"  [Auto-convert] Temperature {temperature} detected as °C → converting to K")
        temp_k = temperature + 273.15
    elif 220 <= temperature <= 330:
        temp_k = temperature  # Already in Kelvin
    else:
        raise ValueError(
            f"Temperature ({temperature}) outside plausible range. "
            f"Expected: -50 to 60 (°C) or 220 to 330 (K)"
        )

    # --- Pressure Check ---
    if 800 <= pressure <= 1100:
        print(f"  [Auto-convert] Pressure {pressure} detected as hPa → converting to Pa")
        pressure_pa = pressure * 100
    elif 80000 <= pressure <= 110000:
        pressure_pa = pressure  # Already in Pa
    else:
        raise ValueError(
            f"Pressure ({pressure}) outside plausible range. "
            f"Expected: 800 to 1100 (hPa) or 80000 to 110000 (Pa)"
        )
        
    # --- Volume Check ---
    if 1 <= volume <= 1000:
        print(f"  [Auto-convert] Volume {volume} detected as Liters → converting to m³")
        volume_m3 = volume / 1000.0
    elif 0.001 <= volume <= 1:
        volume_m3 = volume  # Already in m³
    else:
        raise ValueError(
            f"Volume ({volume}) outside plausible range. "
            f"Expected: 1 to 1000 (L) or 0.001 to 1 (m³)"
        )

    # --- Area Check ---
    if 100 <= area <= 10000:
        print(f"  [Auto-convert] Area {area} detected as cm² → converting to m²")
        area_m2 = area / 10000.0
    elif 0.01 <= area <= 1:
        area_m2 = area  # Already in m²
    else:
        raise ValueError(
            f"Area ({area}) outside plausible range. "
            f"Expected: 100 to 10000 (cm²) or 0.01 to 1 (m²)"
        )

    # --- Core Calculation ---
    slope_ppm_s = slope_ppb_s / 1000.0
    gas_moles = (pressure_pa * volume_m3) / (R * temp_k)
    flux = slope_ppm_s * gas_moles / area_m2 * 1e6
    
    return flux
```

**Example usage:**

```python
# This will auto-convert units and print messages
flux = calculate_flux_robust(
    slope_ppb_s=0.05,
    temperature=25,      # Celsius - will be converted
    pressure=1013,       # hPa - will be converted  
    volume=12.6,         # Liters - will be converted
    area=1257            # cm² - will be converted
)
print(f"\nFlux: {flux:.5f} µmol m⁻² s⁻¹")
```

</details>

> **Info: Python's `raise` Keyword**
> 
> The `raise` keyword is used to trigger an exception (error) when something goes wrong:
> 
> ```python
> raise ValueError("Your error message here")
> ```
> 
> Common exception types:
> - `ValueError`: Input has wrong value (but correct type)
> - `TypeError`: Input has wrong type
> - `RuntimeError`: General runtime error
>
> **Resources:** [Python raise keyword (GeeksforGeeks)](https://www.geeksforgeeks.org/python-raise-keyword/)

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
