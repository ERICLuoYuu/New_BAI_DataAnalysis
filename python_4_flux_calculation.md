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
# Rename temperature column
df_Ta['Ta_C'] = df_Ta['Temperature , °C']
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
    df_Ta_reset[['Timestamp', 'Ta_C']].sort_values('Timestamp'),
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

# 2. Select where N2O has valid values (Drop NaNs), we also need the temperature column for flux calculation later on
df_N2O = df_merged[['N2O_ppb', 'Ta_C']].dropna() 

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

The flux calculation is a two-step process:

**Step 1: Calculate moles of air in the chamber** using the Ideal Gas Law:

$$
n = \frac{P \cdot V}{R \cdot T}
$$

**Step 2: Calculate flux** by combining the slope with moles of air:

$$
\text{Flux} = \frac{n \cdot \text{slope}}{A}
$$

Where:

| Symbol | Description | Unit |
|--------|-------------|------|
| $n$ | Moles of air in the chamber | mol |
| $P$ | Atmospheric pressure | atm |
| $V$ | Chamber headspace volume | L (liters) |
| $R$ | Ideal gas constant | 0.0821 L·atm·K⁻¹·mol⁻¹ |
| $T$ | Air temperature | K (Kelvin) |
| slope | Rate of concentration change | ppm s⁻¹ |
| $A$ | Surface area covered by chamber | m² |

#### Understanding Why This Works

The key insight is understanding what **ppm** means:

> **ppm = parts per million = µmol per mol of air**

So when we multiply:
- `slope (ppm/s)` × `n (moles of air)` = **µmol/s**

Then dividing by area gives us:
- `µmol/s` ÷ `area (m²)` = **µmol m⁻² s⁻¹**

This is our final flux unit—no additional conversion factor needed!

> **Info: ppb vs ppm - Which Unit to Use?**
> 
> Gas concentrations can be expressed in different units:
> - **ppm** (parts per million) = 1 molecule per 1,000,000 air molecules
> - **ppb** (parts per billion) = 1 molecule per 1,000,000,000 air molecules
> - **1 ppm = 1000 ppb**
> 
> Typical usage by gas type:
> - **CO₂**: Usually measured in **ppm** (atmospheric ~420 ppm)
> - **CH₄**: Can be ppm or ppb (atmospheric ~1.9 ppm = 1900 ppb)
> - **N₂O**: Usually measured in **ppb** (atmospheric ~330 ppb)
> 
> Always check your instrument output to know which unit your slope is in!

> **Info: The Ideal Gas Law**
> 
> The equation $PV = nRT$ relates pressure, volume, and temperature to the number of moles of gas.
> 
> Rearranging: $n = \frac{PV}{RT}$
> 
> **Important:** Be careful with units! If using:
> - $R = 0.0821$ L·atm·K⁻¹·mol⁻¹ → use $V$ in liters, $P$ in atm
> - $R = 8.314$ J·K⁻¹·mol⁻¹ → use $V$ in m³, $P$ in Pa
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
- Handle both ppb/s and ppm/s slope units using a parameter
- Use the Ideal Gas Law to calculate moles: $n = \frac{PV}{RT}$
- Remember: ppm means µmol per mol, so `slope_ppm × n` already gives µmol!

```python
# Define the ideal gas constant (using L, atm, K, mol units)
R = 0.0821  # L·atm·K⁻¹·mol⁻¹

def calculate_flux(slope, temp_k, pressure_atm, volume_L, area_m2, slope_unit='ppb'):
    """
    Calculates gas flux from chamber measurements.

    Parameters:
    - slope (float): Rate of concentration change
    - temp_k (float): Temperature in Kelvin
    - pressure_atm (float): Pressure in atmospheres (typically 1 atm)
    - volume_L (float): Chamber volume in liters
    - area_m2 (float): Chamber area in square meters
    - slope_unit (str): Unit of slope - 'ppb' for ppb/s or 'ppm' for ppm/s
    
    Returns:
    - float: Flux in µmol m⁻² s⁻¹
    """
    # Step 1: Convert slope to ppm/s if needed
    if slope_unit == 'ppb':
        slope_ppm_s = ...
    elif slope_unit == 'ppm':
        slope_ppm_s = ...
    else:
        raise ValueError(...)
    
    # Step 2: Calculate moles of air using Ideal Gas Law (n = PV/RT)
    n_moles = ...
    
    # Step 3: Calculate flux in µmol m⁻² s⁻¹
    # Note: ppm = µmol/mol, so (slope_ppm × n_moles) gives µmol/s
    flux = ...
    
    return flux
```

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# Define the ideal gas constant (using L, atm, K, mol units)
R = 0.0821  # L·atm·K⁻¹·mol⁻¹

def calculate_flux(slope, temp_k, pressure_atm, volume_L, area_m2, slope_unit='ppb'):
    """
    Calculates gas flux from chamber measurements.

    Parameters:
    - slope (float): Rate of concentration change
    - temp_k (float): Temperature in Kelvin
    - pressure_atm (float): Pressure in atmospheres (typically 1 atm)
    - volume_L (float): Chamber volume in liters
    - area_m2 (float): Chamber area in square meters
    - slope_unit (str): Unit of slope - 'ppb' for ppb/s or 'ppm' for ppm/s
    
    Returns:
    - float: Flux in µmol m⁻² s⁻¹
    """
    # Step 1: Convert slope to ppm/s if needed
    if slope_unit == 'ppb':
        slope_ppm_s = slope / 1000.0  # Convert ppb to ppm
    elif slope_unit == 'ppm':
        slope_ppm_s = slope  # Already in ppm
    else:
        raise ValueError(f"slope_unit must be 'ppb' or 'ppm', got '{slope_unit}'")
    
    # Step 2: Calculate moles of air using Ideal Gas Law (n = PV/RT)
    n_moles = (pressure_atm * volume_L) / (R * temp_k)
    
    # Step 3: Calculate flux in µmol m⁻² s⁻¹
    # Note: ppm = µmol/mol, so (slope_ppm × n_moles) gives µmol/s
    # Dividing by area gives µmol/m²/s - NO additional conversion needed!
    flux = (n_moles * slope_ppm_s) / area_m2
    
    return flux
```

**Example calculations:**

```python
# Example 1: Using slope in ppb/s (typical for N2O)
flux_n2o = calculate_flux(
    slope=0.05,           # 0.05 ppb/s
    temp_k=298.15,
    pressure_atm=1.0,
    volume_L=12.6,
    area_m2=0.1257,
    slope_unit='ppb'      # Specify unit
)
print(f"N2O Flux: {flux_n2o:.6f} µmol m⁻² s⁻¹")

# Example 2: Using slope in ppm/s (typical for CO2)
flux_co2 = calculate_flux(
    slope=0.0842,         # 0.0842 ppm/s
    temp_k=306.11,
    pressure_atm=1.0,
    volume_L=41.46,
    area_m2=0.123,
    slope_unit='ppm'      # Specify unit
)
print(f"CO2 Flux: {flux_co2:.4f} µmol m⁻² s⁻¹")
# Expected: ~1.13 µmol m⁻² s⁻¹
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>


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

> **⚠️ Critical: Use Only the Linear Phase**
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


### 3.5 Final Flux Calculation

Now we have all the pieces. Let's put them together!

#### Step 1: Get Average Temperature and Pressure

We need the mean temperature and pressure during the measurement. **Unit conversion is critical!**

```python
# Get average temperature (convert °C to Kelvin)
avg_temp_c = regression_data['Ta_C'].mean()
avg_temp_k = avg_temp_c + 273.15

# For pressure, we assume standard atmospheric pressure
# (If you have measured pressure data, use that instead)
pressure_atm = 1.0  # atm

print(f"Average Temperature: {avg_temp_c:.2f} °C = {avg_temp_k:.2f} K")
print(f"Pressure: {pressure_atm} atm")
```

#### Step 2: Define Chamber Dimensions

The chamber volume and area are constants for our setup. **Make sure volume is in liters!**

```python
# Chamber specifications (measure these for your specific equipment!)
CHAMBER_VOLUME_L = 41.4567    # liters
COLLAR_AREA_M2 = 0.123    # m²

print(f"Chamber Volume: {CHAMBER_VOLUME_L} L")
print(f"Collar Area: {COLLAR_AREA_M2} m²")
```

#### Step 3: Calculate Moles of Air

Using the Ideal Gas Law:

```python
# Ideal gas constant (L·atm·K⁻¹·mol⁻¹)
R = 0.0821

# Calculate moles of air in the chamber
n_moles = (pressure_atm * CHAMBER_VOLUME_L) / (R * avg_temp_k)

print(f"Moles of air in chamber: {n_moles:.4f} mol")
```

#### Step 4: Calculate the Flux

```python
# Convert slope from ppb/s to ppm/s
slope_ppm_s = slope / 1000.0

# Calculate flux!
# Remember: ppm = µmol/mol, so (n × slope_ppm) gives µmol/s
flux_n2o = (n_moles * slope_ppm_s) / COLLAR_AREA_M2

print("\n" + "="*50)
print("        FINAL FLUX CALCULATION RESULT")
print("="*50)
print(f"  Slope:       {slope:.4f} ppb/s = {slope_ppm_s:.6f} ppm/s")
print(f"  Temperature: {avg_temp_k:.2f} K")
print(f"  Pressure:    {pressure_atm} atm")
print(f"  Volume:      {CHAMBER_VOLUME_L} L")
print(f"  Moles (n):   {n_moles:.4f} mol")
print(f"  Area:        {COLLAR_AREA_M2} m²")
print("-"*50)
print(f"  N₂O Flux:    {flux_n2o:.5f} µmol m⁻² s⁻¹")
print("="*50)
```

Or use our function:

```python
flux_n2o = calculate_flux(
    slope=slope,
    temp_k=avg_temp_k,
    pressure_atm=pressure_atm,
    volume_L=CHAMBER_VOLUME_L,
    area_m2=COLLAR_AREA_M2,
    slope_unit='ppb'  # Our regression slope is in ppb/s
)

print(f"N₂O Flux: {flux_n2o:.5f} µmol m⁻² s⁻¹")
```

**Congratulations!** You've successfully converted raw gas concentration data into a standardized flux value!


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
| Pressure | 0.8 to 1.2 | atm |
| Pressure | 800 to 1200 | hPa/mbar |
| Volume | 1 to 1000 | Liters |
| Volume | 0.001 to 1 | m³ |
| Area | 100 to 10,000 | cm² |
| Area | 0.01 to 1 | m² |

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# Ideal gas constant (using L, atm, K, mol units)
R = 0.0821  # L·atm·K⁻¹·mol⁻¹

def calculate_flux_robust(slope, temperature, pressure, volume, area, slope_unit='ppb'):
    """
    Calculates gas flux with automatic unit detection and conversion.
    
    All inputs are auto-converted to standard units:
    - Temperature → Kelvin
    - Pressure → atm
    - Volume → Liters
    - Area → m²

    Parameters:
    - slope (float): Rate of concentration change
    - temperature (float): Temperature in Celsius or Kelvin (auto-detected)
    - pressure (float): Pressure in atm or hPa (auto-detected)
    - volume (float): Chamber volume in m³ or Liters (auto-detected)
    - area (float): Chamber area in m² or cm² (auto-detected)
    - slope_unit (str): Unit of slope - 'ppb' for ppb/s or 'ppm' for ppm/s
    
    Returns:
    - float: Flux in µmol m⁻² s⁻¹
    """
    
    # --- Slope Unit Check ---
    if slope_unit == 'ppb':
        slope_ppm_s = slope / 1000.0
    elif slope_unit == 'ppm':
        slope_ppm_s = slope
    else:
        raise ValueError(f"slope_unit must be 'ppb' or 'ppm', got '{slope_unit}'")
    
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
    if 0.8 <= pressure <= 1.2:
        pressure_atm = pressure  # Already in atm
    elif 800 <= pressure <= 1200:
        print(f"  [Auto-convert] Pressure {pressure} detected as hPa → converting to atm")
        pressure_atm = pressure / 1013.25  # Convert hPa to atm
    else:
        raise ValueError(
            f"Pressure ({pressure}) outside plausible range. "
            f"Expected: 0.8 to 1.2 (atm) or 800 to 1200 (hPa)"
        )
        
    # --- Volume Check ---
    if 1 <= volume <= 1000:
        volume_L = volume  # Already in Liters
    elif 0.001 <= volume <= 1:
        print(f"  [Auto-convert] Volume {volume} detected as m³ → converting to L")
        volume_L = volume * 1000.0  # Convert m³ to L
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
    # Calculate moles of air (n = PV/RT)
    n_moles = (pressure_atm * volume_L) / (R * temp_k)
    
    # Calculate flux
    # ppm = µmol/mol, so (n × slope_ppm) gives µmol/s
    # Dividing by area gives µmol/m²/s
    flux = (n_moles * slope_ppm_s) / area_m2
    
    return flux
```

**Example usage:**

```python
# Example 1: N2O with slope in ppb/s
flux_n2o = calculate_flux_robust(
    slope=50,             # 50 ppb/s
    temperature=25,       # Celsius - will be converted to K
    pressure=1013,        # hPa - will be converted to atm
    volume=12.6,          # Liters - already correct
    area=1257,            # cm² - will be converted to m²
    slope_unit='ppb'
)
print(f"N2O Flux: {flux_n2o:.5f} µmol m⁻² s⁻¹")

# Example 2: CO2 with slope in ppm/s
flux_co2 = calculate_flux_robust(
    slope=0.0842,         # 0.0842 ppm/s
    temperature=33,       # Celsius
    pressure=1.0,         # atm - already correct
    volume=41.46,         # Liters
    area=0.123,           # m² - already correct
    slope_unit='ppm'
)
print(f"CO2 Flux: {flux_co2:.4f} µmol m⁻² s⁻¹")
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

## 4. Automating Gas Flux Calculation

In Section 3, we calculated flux for a single measurement by hand. Now it's time to scale up! With dozens of measurements across multiple plots, dates, and gas types, we need automation.

### 4.1 Structuring Measurement Metadata
The first and crucial step of automation is to store the key information (metadata) for each measurement in a structured way that a program can loop through. For this, we will use a Python **dictionary** that can be converted to a DataFrame. The dictionary keys will be our data “columns” (e.g., ‘plot_id’, ‘land_use’), and the values will be lists containing the data for each plot.  
### The Challenge: Multiple Measurements per Plot

Now, there is an issue: we take multiple measurements at the same plot, perhaps on different days or at different times. How can we store this information efficiently?

**Solution:** We can store the multiple start and end times for a single **plot** as a single string, with each timestamp separated by a semicolon (;). Of course, the order of the multiple starttime and endtime for a plot should match. By doing this, we can keep the metadata table concise and still tell our program to perform multiple calculations for that plot:
- `plot_id`: The unique plot identifier
- `land_use`: The land cover type
- `start_time`: All measurement start times, separated by `;`
- `end_time`: All measurement end times, separated by `;`
- `variable`: The gas measured for each time window, separated by `;`

> **Important:** The order must match! The 1st start_time goes with the 1st end_time and 1st variable, etc.

### Creating the Metadata Dictionary

Now let's create our metadata. A Python dictionary uses **keys** (column names) and **values** (lists of data):

```python
import pandas as pd

# --- Create Compact Measurement Metadata ---
# Each plot appears only once; multiple measurements are semicolon-separated

measurement_info = {
    'plot_id': ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3'],
    'land_use': ['forest', 'forest', 'forest', 'grassland', 'grassland', 'grassland'],
    'start_time': [
        '2025-08-06 11:41:35; 2025-08-06 11:41:35; 2025-08-15 11:08:25; 2025-08-15 11:08:25; 2025-08-15 11:09:36; 2025-08-26 11:12:39; 2025-08-26 11:12:39; 2025-08-26 11:14:16',
        '2025-08-06 11:52:00; 2025-08-06 11:52:00; 2025-08-15 11:20:30; 2025-08-15 11:20:30; 2025-08-15 11:21:41; 2025-08-26 11:18:35; 2025-08-26 11:18:35; 2025-08-26 11:20:10',
        '2025-08-26 11:24:51; 2025-08-26 11:24:51; 2025-08-26 11:26:35',
        '2025-08-06 12:11:13; 2025-08-06 12:11:13; 2025-08-15 12:11:38; 2025-08-15 12:11:38; 2025-08-15 12:12:49; 2025-08-26 11:49:07; 2025-08-26 11:50:40; 2025-08-26 11:50:40',
        '2025-08-06 12:21:00; 2025-08-06 12:21:00; 2025-08-15 12:03:45; 2025-08-15 12:03:45; 2025-08-15 12:04:56; 2025-08-26 12:04:23; 2025-08-26 12:04:23; 2025-08-26 12:05:31',
        '2025-08-06 12:27:47; 2025-08-06 12:27:47; 2025-08-15 11:53:30; 2025-08-15 11:53:30; 2025-08-15 11:54:41; 2025-08-26 12:25:48; 2025-08-26 12:25:48; 2025-08-26 12:27:23'
    ],
    'end_time': [
        '2025-08-06 11:45:35; 2025-08-06 11:45:35; 2025-08-15 11:12:25; 2025-08-15 11:12:25; 2025-08-15 11:13:36; 2025-08-26 11:16:39; 2025-08-26 11:16:39; 2025-08-26 11:18:16',
        '2025-08-06 11:56:00; 2025-08-06 11:56:00; 2025-08-15 11:24:30; 2025-08-15 11:24:30; 2025-08-15 11:25:41; 2025-08-26 11:22:35; 2025-08-26 11:22:35; 2025-08-26 11:24:10',
        '2025-08-26 11:28:51; 2025-08-26 11:28:51; 2025-08-26 11:30:35',
        '2025-08-06 12:17:00; 2025-08-06 12:17:00; 2025-08-15 12:15:38; 2025-08-15 12:15:38; 2025-08-15 12:16:49; 2025-08-26 11:53:07; 2025-08-26 11:54:40; 2025-08-26 11:54:40',
        '2025-08-06 12:25:46; 2025-08-06 12:25:46; 2025-08-15 12:07:45; 2025-08-15 12:07:45; 2025-08-15 12:08:56; 2025-08-26 12:08:23; 2025-08-26 12:08:23; 2025-08-26 12:09:31',
        '2025-08-06 12:31:50; 2025-08-06 12:31:50; 2025-08-15 11:57:59; 2025-08-15 11:57:59; 2025-08-15 11:59:10; 2025-08-26 12:29:48; 2025-08-26 12:29:48; 2025-08-26 12:31:23'
    ],
    'variable': [
        'CH4; CO2; CH4; CO2; N2O; CH4; CO2; N2O',
        'CH4; CO2; CH4; CO2; N2O; CH4; CO2; N2O',
        'CH4; CO2; N2O',
        'CH4; CO2; CH4; CO2; N2O; N2O; CH4; CO2',
        'CH4; CO2; CH4; CO2; N2O; CH4; CO2; N2O',
        'CH4; CO2; CH4; CO2; N2O; CH4; CO2; N2O'
    ]
}

# Convert dictionary to DataFrame
metadata_df = pd.DataFrame(measurement_info)
```

Let's verify our metadata:

```python
print("Measurement Metadata Summary:")
print(f"  Total plots: {len(metadata_df)}")
print(f"  Forest plots: {(metadata_df['land_use'] == 'forest').sum()}")
print(f"  Grassland plots: {(metadata_df['land_use'] == 'grassland').sum()}")

# Count total measurements across all plots
total_measurements = sum(len(row['variable'].split(';')) for _, row in metadata_df.iterrows())
print(f"  Total measurements: {total_measurements}")

metadata_df
```

### How to Parse Semicolon-Separated Values

Now we need to learn how to **split** these combined strings back into individual values. Python provides the `split()` method for this.

> **Info: The `split()` Method**
> 
> The `split()` method breaks a string into a list of substrings based on a delimiter (separator).
> 
> ```python
> # Basic usage
> text = "apple; banana; cherry"
> fruits = text.split('; ')  # Split by '; '
> print(fruits)  # ['apple', 'banana', 'cherry']
> 
> # Access individual items
> print(fruits[0])  # 'apple'
> print(fruits[1])  # 'banana'
> ```
> 
> **Common delimiters:**
> - `'; '` — semicolon with space (our format)
> - `','` — comma
> - `' '` — space
> - `'\t'` — tab
> - `'\n'` — newline
>
> **Resources:** [Python split() documentation](https://docs.python.org/3/library/stdtypes.html#str.split)

Let's practice parsing one plot's data:

```python
# Get data for Plot 1-1
plot_data = metadata_df[metadata_df['plot_id'] == '1-1'].iloc[0]

print(f"Plot: {plot_data['plot_id']}")
print(f"Land use: {plot_data['land_use']}")

# The raw string looks like this:
print(f"\nRaw variable string: {plot_data['variable']}")

# Split it into a list
variables = plot_data['variable'].split('; ')
print(f"\nAfter split(): {variables}")
print(f"Number of measurements: {len(variables)}")
```

Output:
```
Plot: 1-1
Land use: forest

Raw variable string: CH4; CO2; CH4; CO2; N2O; CH4; CO2; N2O

After split(): ['CH4', 'CO2', 'CH4', 'CO2', 'N2O', 'CH4', 'CO2', 'N2O']
Number of measurements: 8
```

### Iterating Through Parallel Lists with `zip()`

We now have three lists that need to be processed together:
- `start_times` — when each measurement started
- `end_times` — when each measurement ended  
- `variables` — which gas was measured

These lists are **parallel**: the 1st item in each list belongs together, the 2nd items belong together, etc. How do we loop through them simultaneously?

> **Info: The `zip()` Function**
> 
> `zip()` takes multiple lists and combines them element-by-element, like a zipper bringing two sides together:
> 
> ```python
> names = ['Alice', 'Bob', 'Charlie']
> ages = [25, 30, 35]
> cities = ['NYC', 'LA', 'Chicago']
> 
> # Without zip - awkward index-based loop
> for i in range(len(names)):
>     print(f"{names[i]} is {ages[i]} from {cities[i]}")
> 
> # With zip - clean and readable
> for name, age, city in zip(names, ages, cities):
>     print(f"{name} is {age} from {city}")
> ```
> 
> Output:
> ```
> Alice is 25 from NYC
> Bob is 30 from LA
> Charlie is 35 from Chicago
> ```
> 
> **Why use `zip()`?**
> - Cleaner, more readable code
> - No need to manage index variables
> - Less prone to off-by-one errors
> - Pythonic way to handle parallel iteration
>
> **Resources:** [Python zip() documentation](https://docs.python.org/3/library/functions.html#zip)

Let's use `zip()` to display all measurements for Plot 1-1:

```python
# Parse all three columns
start_times = plot_data['start_time'].split('; ')
end_times = plot_data['end_time'].split('; ')
variables = plot_data['variable'].split('; ')

print(f"Plot {plot_data['plot_id']} ({plot_data['land_use']}):")
print(f"Total measurements: {len(variables)}\n")

# Use zip() to iterate through all three lists together
for i, (start, end, var) in enumerate(zip(start_times, end_times, variables), 1):
    print(f"  {i}. {var}: {start} → {end}")
```

> **Note:** We also use `enumerate(..., 1)` to get a counter starting from 1. This is another useful Python pattern for numbered loops.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise: Parse and Display Another Plot

Now it's your turn! Write code to parse and display measurements for Plot `2-1` (a grassland plot).

**Your code should:**
1. Extract the row for plot_id == '2-1'
2. Split the start_time, end_time, and variable strings
3. Use `zip()` to print each measurement with its number

**Bonus questions to answer:**
- How many measurements does Plot 2-1 have?
- Which unique gases were measured? (Hint: use `set()`)
- On which dates were measurements taken?

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# Get data for Plot 2-1
plot_2_1 = metadata_df[metadata_df['plot_id'] == '2-1'].iloc[0]

# Parse the semicolon-separated values
start_times = plot_2_1['start_time'].split('; ')
end_times = plot_2_1['end_time'].split('; ')
variables = plot_2_1['variable'].split('; ')

print(f"Plot: {plot_2_1['plot_id']} ({plot_2_1['land_use']})")
print(f"Number of measurements: {len(variables)}")

# Find unique gases using set()
unique_gases = set(variables)
print(f"Gases measured: {unique_gases}")

# Find unique dates (extract date part from timestamps)
unique_dates = set(s.split()[0] for s in start_times)
print(f"Measurement dates: {sorted(unique_dates)}")

print(f"\nAll measurements:")
for i, (start, end, var) in enumerate(zip(start_times, end_times, variables), 1):
    print(f"  {i}. {var}: {start} → {end}")
```

**Answers:**
- Plot 2-1 has **8 measurements**
- Gases: **{'CH4', 'CO2', 'N2O'}**
- Dates: **['2025-08-06', '2025-08-15', '2025-08-26']**
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

### 4.2 Setting Up Configuration and Constants

### Why Use a Configuration Dictionary?

Different gases have different properties:
- **N₂O** concentrations are in **ppb** (parts per billion)
- **CO₂** and **CH₄** concentrations are in **ppm** (parts per million)
- Each gas is stored in a different column of our DataFrame

We *could* write separate code for each gas:
```python
# Bad approach - lots of repetition!
if variable == 'N2O':
    df = df_n2o
    column = 'N2O_ppb'
    unit = 'ppb'
elif variable == 'CO2':
    df = df_co2
    column = 'CO2_ppm'
    unit = 'ppm'
elif variable == 'CH4':
    # ... and so on
```

But this is repetitive and hard to maintain. What if we add a new gas? We'd have to update multiple places.

**Better approach:** Store all gas-specific settings in a **configuration dictionary**. Then our code can look up any gas's settings dynamically.

> **Info: Nested Dictionaries**
> 
> A dictionary can contain other dictionaries as values, creating a nested structure:
> 
> ```python
> # Simple dictionary
> person = {'name': 'Alice', 'age': 30}
> 
> # Nested dictionary
> people = {
>     'alice': {'name': 'Alice', 'age': 30, 'city': 'NYC'},
>     'bob': {'name': 'Bob', 'age': 25, 'city': 'LA'}
> }
> 
> # Accessing nested values
> print(people['alice']['city'])  # 'NYC'
> print(people['bob']['age'])     # 25
> ```
> 
> This is perfect for storing configuration where each "category" (gas type) has multiple properties.

### Creating the Gas Configuration

```python
# --- Gas Configuration Dictionary ---
# Maps each gas name to its specific settings

GAS_CONFIG = {
    'N2O': {
        'dataframe': df_merged[['N2O_ppb', 'Ta_C']].dropna(),
        'column': 'N2O_ppb',
        'slope_unit': 'ppb',
        'display_name': 'N₂O'
    },
    'CO2': {
        'dataframe': df_merged[['CO2_ppm', 'Ta_C']].dropna(),
        'column': 'CO2_ppm',
        'slope_unit': 'ppm',
        'display_name': 'CO₂'
    },
    'CH4': {
        'dataframe': df_merged[['CH4_ppm', 'Ta_C']].dropna(),
        'column': 'CH4_ppm',
        'slope_unit': 'ppm',
        'display_name': 'CH₄'
    }
}
```

Now we can access any gas's settings with a simple lookup:

```python
# Example: Get settings for CO2
gas_name = 'CO2'
config = GAS_CONFIG[gas_name]

print(f"Settings for {gas_name}:")
print(f"  Column name: {config['column']}")
print(f"  Slope unit: {config['slope_unit']}")
print(f"  Display name: {config['display_name']}")
print(f"  Data points available: {len(config['dataframe'])}")
```

**Benefits of this approach:**
- **DRY (Don't Repeat Yourself):** One loop handles all gases
- **Extensible:** Adding a new gas = adding one dictionary entry
- **Maintainable:** All settings are in one place

### Setting Up Chamber Constants

Our flux calculation also needs physical constants about our measurement chamber. Let's define these clearly:

```python
# --- Chamber Physical Constants ---
# Measure these for YOUR specific equipment!

CHAMBER_VOLUME_L = 41.4567    # Internal volume in liters
COLLAR_AREA_M2 = 0.123        # Ground area covered in square meters
PRESSURE_ATM = 1.0            # Atmospheric pressure (assume standard)

print("Chamber Configuration:")
print(f"  Volume: {CHAMBER_VOLUME_L} L")
print(f"  Collar Area: {COLLAR_AREA_M2} m²")
print(f"  Pressure: {PRESSURE_ATM} atm")
```

### Setting Up Quality Control Thresholds

Not every measurement will produce a valid flux. Sometimes the chamber leaks, or the soil flux is too small to detect, or there's instrument noise. We need **quality control (QC) thresholds** to identify good vs. bad measurements.

```python
# --- Quality Control Thresholds ---
R_SQUARED_THRESHOLD = 0.70    # Minimum R² (coefficient of determination)
P_VALUE_THRESHOLD = 0.05      # Maximum p-value for statistical significance

print("Quality Control Thresholds:")
print(f"  Minimum R²: {R_SQUARED_THRESHOLD} (regression must explain ≥70% of variance)")
print(f"  Maximum p-value: {P_VALUE_THRESHOLD} (slope must be statistically significant)")
```

**What do these thresholds mean?**
- **R² ≥ 0.70:** The linear fit must explain at least 70% of the variation in concentration
- **p-value ≤ 0.05:** The slope must be statistically significant (not due to random chance)

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise: Explore the Configuration

Write code to:
1. Loop through all gases in `GAS_CONFIG` and print their settings
2. Find which gas has the most data points available
3. Verify that we have data for all three measurement dates

**Hint:** Use `for gas_name, config in GAS_CONFIG.items():` to iterate through dictionary items.

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
print("="*50)
print("GAS CONFIGURATION SUMMARY")
print("="*50)

max_points = 0
gas_with_most_data = None

for gas_name, config in GAS_CONFIG.items():
    n_points = len(config['dataframe'])
    
    print(f"\n{config['display_name']} ({gas_name}):")
    print(f"  Column: {config['column']}")
    print(f"  Unit: {config['slope_unit']}")
    print(f"  Data points: {n_points:,}")
    
    # Check date range
    df = config['dataframe']
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Track which has most data
    if n_points > max_points:
        max_points = n_points
        gas_with_most_data = gas_name

print(f"\n→ {gas_with_most_data} has the most data points: {max_points:,}")
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>


### 4.3 Building the Automation Pipeline

Now we're ready to build the main automation loop. This is the most complex part of the tutorial, so we'll break it into clear steps and explain each one thoroughly.

### Overview: What Will Our Pipeline Do?

Here's the high-level workflow for processing **each measurement**:
```
For each PLOT in metadata:
    │
    └── For each MEASUREMENT in that plot:
            │
            ├── 1️⃣ Extract data for the time window
            │
            ├── 2️⃣ Show plot for visual inspection
            │
            ├── 3️⃣ Let user refine the time window
            │
            ├── 4️⃣ Perform linear regression
            │
            ├── 5️⃣ Check quality (R², p-value)
            │       │
            │       ├── ✓ Passed → Calculate flux
            │       └── ✗ Failed → Mark as invalid
            │
            └── 6️⃣ Store results
```

This is a **nested loop** structure:
- **Outer loop:** Goes through each plot (6 plots)
- **Inner loop:** Goes through each measurement within that plot (varies per plot)

### Step 1: Create Storage for Results

Before we start processing, we need a place to store all our calculated fluxes. We'll use a **list of dictionaries** that can later be converted to a DataFrame.

**Why a list of dictionaries?**

```python
# Each measurement's results will be a dictionary
result = {
    'plot_id': '1-1',
    'variable': 'CO2',
    'flux': 1.234,
    # ... more fields
}

# We append each result to a list
results = []
results.append(result)

# At the end, convert to DataFrame
results_df = pd.DataFrame(results)
```

This pattern is very common in Python data processing!

```python
from scipy import stats
import numpy as np

# --- Initialize Results Storage ---
results = []  # Will hold one dictionary per measurement
```

### Step 2: Create a Helper Function for Temperature

During flux calculation, we need the average temperature during each measurement. Let's create a reusable function for this:

```python
def get_mean_temperature(start_time, end_time, temp_df, temp_column='Ta_C'):
    """
    Calculate mean temperature during a measurement window.
    
    Parameters:
        start_time: Start of measurement (datetime)
        end_time: End of measurement (datetime)
        temp_df: DataFrame with temperature data (DatetimeIndex)
        temp_column: Name of temperature column
    
    Returns:
        float: Mean temperature in °C, or NaN if no data found
    """
    # Create a boolean mask for the time window
    mask = (temp_df.index >= start_time) & (temp_df.index <= end_time)
    
    # Extract temperature values in that window
    temp_values = temp_df.loc[mask, temp_column]
    
    # Handle case where no data is found
    if len(temp_values) == 0:
        print(f"  ⚠ Warning: No temperature data found")
        return np.nan
    
    return temp_values.mean()
```

### Step 3: Count Total Measurements

For progress tracking, let's count how many measurements we'll process:

```python
# Count total measurements across all plots
total_measurements = 0
for _, row in metadata_df.iterrows():
    n_measurements = len(row['variable'].split(';'))
    total_measurements += n_measurements

print(f"Total measurements to process: {total_measurements}")
```

Or more concisely using a **generator expression**:

```python
total_measurements = sum(
    len(row['variable'].split(';')) 
    for _, row in metadata_df.iterrows()
)
print(f"Total measurements to process: {total_measurements}")
```

### Step 4: Understanding the Nested Loop Structure

Before writing the full loop, let's understand its structure with a simplified version that just prints what it would do:

> **Info: Nested Loops**
> 
> A nested loop is a loop inside another loop. The inner loop runs completely for each iteration of the outer loop.
> 
> ```python
> # Example: Print a multiplication table
> for i in range(1, 4):        # Outer loop: i = 1, 2, 3
>     for j in range(1, 4):    # Inner loop: j = 1, 2, 3
>         print(f"{i} × {j} = {i*j}")
>     print("---")  # Runs after inner loop completes
> ```
> 
> Output:
> ```
> 1 × 1 = 1
> 1 × 2 = 2
> 1 × 3 = 3
> ---
> 2 × 1 = 2
> 2 × 2 = 4
> ...
> ```
> 
> In our case:
> - **Outer loop:** Each plot
> - **Inner loop:** Each measurement within that plot

```python
# --- Preview the loop structure (no calculations) ---
print("="*60)
print("LOOP STRUCTURE PREVIEW")
print("="*60)

measurement_counter = 0

# OUTER LOOP: Each plot
for plot_idx, row in metadata_df.iterrows():
    plot_id = row['plot_id']
    land_use = row['land_use']
    
    # Parse the semicolon-separated values
    variables = [v.strip() for v in row['variable'].split(';')]
    
    print(f"\nPlot {plot_id} ({land_use}): {len(variables)} measurements")
    
    # INNER LOOP: Each measurement for this plot
    for var in variables:
        measurement_counter += 1
        print(f"  [{measurement_counter}] {var}")

print(f"\n→ Total: {measurement_counter} measurements")
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise: Understand the Loop Structure

Before running the full automation, let's make sure you understand the nested loop structure.

**Task:** Modify the preview loop above to also count:
1. How many measurements per gas type (N2O, CO2, CH4)
2. How many measurements per land use (forest, grassland)

**Hint:** Create dictionaries like `gas_counts = {'N2O': 0, 'CO2': 0, 'CH4': 0}` and increment them inside the loop.

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# Initialize counters
gas_counts = {'N2O': 0, 'CO2': 0, 'CH4': 0}
landuse_counts = {'forest': 0, 'grassland': 0}

# Loop through all measurements
for plot_idx, row in metadata_df.iterrows():
    land_use = row['land_use']
    variables = [v.strip() for v in row['variable'].split(';')]
    
    for var in variables:
        # Count by gas
        if var in gas_counts:
            gas_counts[var] += 1
        
        # Count by land use
        landuse_counts[land_use] += 1

print("Measurements by Gas Type:")
for gas, count in gas_counts.items():
    print(f"  {gas}: {count}")

print(f"\nMeasurements by Land Use:")
for lu, count in landuse_counts.items():
    print(f"  {lu}: {count}")

print(f"\nTotal: {sum(gas_counts.values())}")
```

**Expected Output:**
```
Measurements by Gas Type:
  N2O: 11
  CO2: 16
  CH4: 16

Measurements by Land Use:
  forest: 19
  grassland: 24

Total: 43
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>

### Step 5: The Outer Loop - Iterating Through Plots

Now let's start building the real loop. The outer loop uses `iterrows()` to go through each row (plot) in our metadata DataFrame:

> **Info: DataFrame `iterrows()`**
> 
> `iterrows()` lets you loop through a DataFrame row by row:
> 
> ```python
> for index, row in df.iterrows():
>     # index = row number (or index label)
>     # row = a Series containing that row's data
>     print(row['column_name'])
> ```
> 
> Each `row` acts like a dictionary—access columns with `row['column_name']`.

```python
# --- Initialize ---
results = []
measurement_counter = 0

print("="*70)
print("STARTING AUTOMATED FLUX CALCULATION")
print(f"Processing {total_measurements} measurements across {len(metadata_df)} plots")
print("="*70)

# --- OUTER LOOP: Each Plot ---
for plot_idx, row in metadata_df.iterrows():
    
    # Extract plot information
    plot_id = row['plot_id']
    land_use = row['land_use']
    
    # Parse semicolon-separated values into lists
    start_times = [s.strip() for s in row['start_time'].split(';')]
    end_times = [s.strip() for s in row['end_time'].split(';')]
    variables = [v.strip() for v in row['variable'].split(';')]
    
    print(f"\n{'='*70}")
    print(f"PLOT: {plot_id} ({land_use})")
    print(f"Measurements to process: {len(variables)}")
    print("="*70)
    
    # Inner loop will go here...
```

### Step 6: The Inner Loop - Processing Each Measurement

Inside the outer loop, we iterate through each measurement using `zip()`:

```python
    # --- INNER LOOP: Each Measurement for This Plot ---
    for start_str, end_str, variable in zip(start_times, end_times, variables):
        
        measurement_counter += 1
        
        # Convert string timestamps to datetime objects
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        measurement_date = start_time.strftime('%Y-%m-%d')
        
        print(f"\n[{measurement_counter}/{total_measurements}] {variable} | {measurement_date}")
        print("-" * 50)
        
        # Processing steps continue here...
```

### Step 7: Look Up Gas Configuration

For each measurement, we need to get the correct settings from our configuration dictionary:

```python
        # --- Get Gas Configuration ---
        # Check if this gas is in our configuration
        if variable not in GAS_CONFIG:
            print(f"  ✗ ERROR: Unknown gas '{variable}'. Skipping.")
            continue  # Skip to next measurement
        
        # Look up settings for this gas
        config = GAS_CONFIG[variable]
        df_gas = config['dataframe']      # The DataFrame containing this gas's data
        column_name = config['column']    # Column name (e.g., 'CO2_ppm')
        slope_unit = config['slope_unit'] # Unit (e.g., 'ppm')
        display_name = config['display_name']  # Pretty name (e.g., 'CO₂')
```

### Step 8: Extract Data for the Time Window

Now we filter the gas data to only include rows within our measurement time window:

```python
        # --- Extract Data for Time Window ---
        # Create a boolean mask for rows between start and end time
        mask = (df_gas.index >= start_time) & (df_gas.index <= end_time)
        measurement_data = df_gas.loc[mask].copy()
        
        # Check if we have enough data points
        if len(measurement_data) < 10:
            print(f"  ⚠ WARNING: Only {len(measurement_data)} data points. Skipping.")
            
            # Still record this measurement, but mark it as failed
            results.append({
                'plot_id': plot_id,
                'land_use': land_use,
                'variable': variable,
                'measurement_date': measurement_date,
                'qc_pass': False,
                'flux_umol_m2_s': np.nan,
                'note': 'Insufficient data'
            })
            continue  # Skip to next measurement
        
        print(f"  Data points in window: {len(measurement_data)}")
```

### Step 9: Visual Inspection

We display the data so the user can inspect it before regression:

```python
        # --- Visual Inspection ---
        # Plot the data to let the user see the pattern
        plot_time_series(
            measurement_data, 
            y_column=column_name, 
            title=f'{display_name} - Plot {plot_id} - {measurement_date}',
            mode='markers'
        )
```

### Step 10: Interactive Time Window Refinement

We may need to adjust the time window to exclude baseline or drop phases:

```python
        # --- Refine Time Window (Interactive) ---
        print(f"\n  Current window: {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')}")
        print("  Look at the plot and identify the LINEAR accumulation phase.")
        print("  Enter new times to refine, or press Enter to keep current:")
        
        start_input = input(f"    New start time (HH:MM:SS) [{start_time.strftime('%H:%M:%S')}]: ").strip()
        end_input = input(f"    New end time (HH:MM:SS) [{end_time.strftime('%H:%M:%S')}]: ").strip()
        
        # Parse the input (keep original if empty)
        if start_input:
            refined_start = pd.to_datetime(f"{measurement_date} {start_input}")
        else:
            refined_start = start_time
            
        if end_input:
            refined_end = pd.to_datetime(f"{measurement_date} {end_input}")
        else:
            refined_end = end_time
        
        # Re-extract data with refined window
        mask = (df_gas.index >= refined_start) & (df_gas.index <= refined_end)
        regression_data = df_gas.loc[mask].copy()
        
        # Check again if we have enough data
        if len(regression_data) < 10:
            print(f"  ⚠ Refined window has only {len(regression_data)} points. Skipping.")
            results.append({
                'plot_id': plot_id,
                'land_use': land_use,
                'variable': variable,
                'measurement_date': measurement_date,
                'qc_pass': False,
                'flux_umol_m2_s': np.nan,
                'note': 'Insufficient data after refinement'
            })
            continue
```

### Step 11: Perform Linear Regression

Now we fit a line to get the slope (rate of concentration change):

```python
        # --- Linear Regression ---
        # First, convert timestamps to "elapsed seconds" for regression
        regression_data['elapsed_seconds'] = (
            regression_data.index - regression_data.index.min()
        ).total_seconds()
        
        # Perform linear regression using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=regression_data['elapsed_seconds'],
            y=regression_data[column_name]
        )
        
        # Calculate R-squared
        r_squared = r_value ** 2
        
        print(f"\n  Regression Results:")
        print(f"    Slope: {slope:.6f} {slope_unit}/s")
        print(f"    R²: {r_squared:.4f}")
        print(f"    p-value: {p_value:.2e}")
```

### Step 12: Visualize the Regression Fit

We show the data with the fitted line overlaid:

```python
        # --- Visualize Regression Fit ---
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=regression_data['elapsed_seconds'],
            y=regression_data[column_name],
            mode='markers',
            name='Measured Data',
            marker=dict(size=6)
        ))
        
        # Add fitted line
        fig.add_trace(go.Scatter(
            x=regression_data['elapsed_seconds'],
            y=intercept + slope * regression_data['elapsed_seconds'],
            mode='lines',
            name=f'Linear Fit (R²={r_squared:.3f})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{display_name} Regression - Plot {plot_id}',
            xaxis_title='Elapsed Time (seconds)',
            yaxis_title=f'{display_name} Concentration ({slope_unit})',
            template='plotly_white'
        )
        
        fig.show()
```

### Step 13: Quality Control

We check if the regression meets our thresholds:

```python
        # --- Quality Control ---
        if r_squared < R_SQUARED_THRESHOLD or p_value > P_VALUE_THRESHOLD:
            print(f"  ✗ QC FAILED: R²={r_squared:.3f} or p={p_value:.3f} outside thresholds")
            qc_pass = False
            flux = np.nan
            note = f'QC failed: R²={r_squared:.3f}, p={p_value:.3f}'
        else:
            print(f"  ✓ QC PASSED")
            qc_pass = True
            note = ''
            
            # Proceed to flux calculation...
```

### Step 14: Calculate Flux (if QC passed)

If quality control passed, we calculate the flux:

```python
            # --- Calculate Flux ---
            # Get mean temperature during measurement
            avg_temp_c = regression_data['Ta_C'].mean()
            avg_temp_k = avg_temp_c + 273.15  # Convert to Kelvin
            
            # Calculate flux using our function from Section 3
            flux = calculate_flux(
                slope=slope,
                temp_k=avg_temp_k,
                pressure_atm=PRESSURE_ATM,
                volume_L=CHAMBER_VOLUME_L,
                area_m2=COLLAR_AREA_M2,
                slope_unit=slope_unit
            )
            
            print(f"\n  Flux Calculation:")
            print(f"    Temperature: {avg_temp_c:.1f}°C = {avg_temp_k:.1f} K")
            print(f"    {display_name} Flux: {flux:.6f} µmol m⁻² s⁻¹")
```

### Step 15: Store Results

Finally, we save all the information for this measurement:

```python
        # --- Store Results ---
        results.append({
            'plot_id': plot_id,
            'land_use': land_use,
            'variable': variable,
            'measurement_date': measurement_date,
            'start_time': refined_start,
            'end_time': refined_end,
            'slope': slope,
            'slope_unit': slope_unit,
            'r_squared': r_squared,
            'p_value': p_value,
            'qc_pass': qc_pass,
            'flux_umol_m2_s': flux,
            'note': note
        })
```


### 4.4 The Complete Processing Script

Now that we understand each piece, here's the complete script. All the steps above are combined into one runnable block:

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise: Run the Complete Pipeline

Run the complete code below. Process all measurements to get familiar with the pipline and get the result.

**Tips for visual inspection:**
- Look for the **linear accumulation phase** (steady increase)
- Exclude the **baseline** (flat part before chamber closes)
- Exclude the **drop** (sudden decrease when chamber opens)
- If the data looks good, just press Enter to keep the original times

```python
from scipy import stats
import numpy as np
import plotly.graph_objects as go

# ============================================================
# AUTOMATED FLUX CALCULATION - COMPLETE SCRIPT
# ============================================================

# --- Initialize ---
results = []
measurement_counter = 0
total_measurements = sum(len(row['variable'].split(';')) for _, row in metadata_df.iterrows())

print("="*70)
print("AUTOMATED FLUX CALCULATION")
print(f"Processing {total_measurements} measurements across {len(metadata_df)} plots")
print("="*70)

# --- OUTER LOOP: Each Plot ---
for plot_idx, row in metadata_df.iterrows():
    
    plot_id = row['plot_id']
    land_use = row['land_use']
    
    # Parse semicolon-separated values
    start_times = [s.strip() for s in row['start_time'].split(';')]
    end_times = [s.strip() for s in row['end_time'].split(';')]
    variables = [v.strip() for v in row['variable'].split(';')]
    
    print(f"\n{'='*70}")
    print(f"PLOT: {plot_id} ({land_use}) - {len(variables)} measurements")
    print("="*70)
    
    # --- INNER LOOP: Each Measurement ---
    for start_str, end_str, variable in zip(start_times, end_times, variables):
        
        measurement_counter += 1
        start_time = pd.to_datetime(start_str)
        end_time = pd.to_datetime(end_str)
        measurement_date = start_time.strftime('%Y-%m-%d')
        
        print(f"\n[{measurement_counter}/{total_measurements}] {variable} | {measurement_date}")
        print("-" * 50)
        
        # --- Get gas configuration ---
        if variable not in GAS_CONFIG:
            print(f"  ✗ Unknown gas '{variable}'. Skipping.")
            continue
        
        config = GAS_CONFIG[variable]
        df_gas = config['dataframe']
        column_name = config['column']
        slope_unit = config['slope_unit']
        display_name = config['display_name']
        
        # --- Extract data ---
        mask = (df_gas.index >= start_time) & (df_gas.index <= end_time)
        measurement_data = df_gas.loc[mask].copy()
        
        if len(measurement_data) < 10:
            print(f"  ⚠ Only {len(measurement_data)} points. Skipping.")
            results.append({
                'plot_id': plot_id, 'land_use': land_use, 'variable': variable,
                'measurement_date': measurement_date, 'slope': np.nan,
                'slope_unit': slope_unit, 'r_squared': np.nan, 'p_value': np.nan,
                'qc_pass': False, 'flux_umol_m2_s': np.nan, 'note': 'Insufficient data'
            })
            continue
        
        print(f"  Data points: {len(measurement_data)}")
        
        # --- Visual inspection ---
        plot_time_series(measurement_data, y_column=column_name, 
                        title=f'{display_name} - Plot {plot_id} - {measurement_date}',
                        mode='markers')
        
        # --- Refine time window ---
        print(f"\n  Window: {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')}")
        start_input = input(f"    New start (HH:MM:SS) or Enter to keep: ").strip()
        end_input = input(f"    New end (HH:MM:SS) or Enter to keep: ").strip()
        
        refined_start = pd.to_datetime(f"{measurement_date} {start_input}") if start_input else start_time
        refined_end = pd.to_datetime(f"{measurement_date} {end_input}") if end_input else end_time
        
        mask = (df_gas.index >= refined_start) & (df_gas.index <= refined_end)
        regression_data = df_gas.loc[mask].copy()
        
        if len(regression_data) < 10:
            print(f"  ⚠ Refined window too small. Skipping.")
            results.append({
                'plot_id': plot_id, 'land_use': land_use, 'variable': variable,
                'measurement_date': measurement_date, 'slope': np.nan,
                'slope_unit': slope_unit, 'r_squared': np.nan, 'p_value': np.nan,
                'qc_pass': False, 'flux_umol_m2_s': np.nan, 'note': 'Insufficient data after refinement'
            })
            continue
        
        # --- Linear regression ---
        regression_data['elapsed_seconds'] = (
            regression_data.index - regression_data.index.min()
        ).total_seconds()
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=regression_data['elapsed_seconds'],
            y=regression_data[column_name]
        )
        r_squared = r_value ** 2
        
        print(f"\n  Slope: {slope:.6f} {slope_unit}/s | R²: {r_squared:.4f} | p: {p_value:.2e}")
        
        # --- Visualize fit ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=regression_data['elapsed_seconds'], 
                                  y=regression_data[column_name],
                                  mode='markers', name='Data'))
        fig.add_trace(go.Scatter(x=regression_data['elapsed_seconds'],
                                  y=intercept + slope * regression_data['elapsed_seconds'],
                                  mode='lines', name=f'Fit (R²={r_squared:.3f})',
                                  line=dict(color='red')))
        fig.update_layout(title=f'{display_name} Regression - Plot {plot_id}',
                         xaxis_title='Elapsed Time (s)',
                         yaxis_title=f'{display_name} ({slope_unit})',
                         template='plotly_white')
        fig.show()
        
        # --- Quality control ---
        if r_squared < R_SQUARED_THRESHOLD or p_value > P_VALUE_THRESHOLD:
            print(f"  ✗ QC FAILED")
            qc_pass, flux, note = False, np.nan, f'QC failed: R²={r_squared:.3f}'
        else:
            print(f"  ✓ QC PASSED")
            qc_pass, note = True, ''
            
            avg_temp_c = regression_data['Ta_C'].mean()
            avg_temp_k = avg_temp_c + 273.15
            
            flux = calculate_flux(
                slope=slope, temp_k=avg_temp_k, pressure_atm=PRESSURE_ATM,
                volume_L=CHAMBER_VOLUME_L, area_m2=COLLAR_AREA_M2,
                slope_unit=slope_unit
            )
            print(f"  {display_name} Flux: {flux:.6f} µmol m⁻² s⁻¹")
        
        # --- Store results ---
        results.append({
            'plot_id': plot_id, 'land_use': land_use, 'variable': variable,
            'measurement_date': measurement_date, 'slope': slope,
            'slope_unit': slope_unit, 'r_squared': r_squared, 'p_value': p_value,
            'qc_pass': qc_pass, 'flux_umol_m2_s': flux, 'note': note
        })

# --- Convert to DataFrame ---
flux_results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)
print(f"Total processed: {len(flux_results_df)}")
print(f"QC passed: {flux_results_df['qc_pass'].sum()}")
print(f"QC failed: {(~flux_results_df['qc_pass']).sum()}")
```
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>


### 4.5 Analyzing the Results

After processing all measurements, we have a complete DataFrame of results. Let's analyze it!

### Viewing the Results

```python
# Display key columns
display_cols = ['plot_id', 'land_use', 'variable', 'measurement_date', 
                'r_squared', 'qc_pass', 'flux_umol_m2_s']

# Sort by variable, then plot, then date
flux_results_df[display_cols].sort_values(['variable', 'plot_id', 'measurement_date'])
```

### Summary Statistics by Gas Type

```python
print("="*60)
print("FLUX SUMMARY BY GAS TYPE")
print("="*60)

for variable in flux_results_df['variable'].unique():
    # Filter to QC-passed measurements of this gas
    df_var = flux_results_df[
        (flux_results_df['variable'] == variable) & 
        (flux_results_df['qc_pass'] == True)
    ]
    
    if len(df_var) > 0:
        display_name = GAS_CONFIG[variable]['display_name']
        print(f"\n{display_name}:")
        print(f"  Valid measurements: {len(df_var)}")
        print(f"  Mean flux:  {df_var['flux_umol_m2_s'].mean():.6f} µmol m⁻² s⁻¹")
        print(f"  Std dev:    {df_var['flux_umol_m2_s'].std():.6f} µmol m⁻² s⁻¹")
        print(f"  Min:        {df_var['flux_umol_m2_s'].min():.6f} µmol m⁻² s⁻¹")
        print(f"  Max:        {df_var['flux_umol_m2_s'].max():.6f} µmol m⁻² s⁻¹")
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
### Exercise: Investigate Your Results

Using the `flux_results_df` DataFrame, answer these questions:

1. What percentage of measurements passed QC?
2. Which plot had the highest CO₂ flux? On which date?
3. What was the average R² for measurements that passed QC?
4. Are there any patterns in which measurements failed QC?

<details markdown="1">
<summary>Click here for the solution!</summary>

```python
# 1. QC pass rate
total = len(flux_results_df)
passed = flux_results_df['qc_pass'].sum()
print(f"QC Pass Rate: {passed}/{total} = {100*passed/total:.1f}%")

# 2. Highest CO2 flux
co2_valid = flux_results_df[
    (flux_results_df['variable'] == 'CO2') & 
    (flux_results_df['qc_pass'] == True)
]
if len(co2_valid) > 0:
    max_idx = co2_valid['flux_umol_m2_s'].idxmax()
    max_row = co2_valid.loc[max_idx]
    print(f"\nHighest CO₂ flux:")
    print(f"  Plot: {max_row['plot_id']}")
    print(f"  Date: {max_row['measurement_date']}")
    print(f"  Flux: {max_row['flux_umol_m2_s']:.4f} µmol m⁻² s⁻¹")

# 3. Average R² for passed measurements
passed_df = flux_results_df[flux_results_df['qc_pass'] == True]
print(f"\nAverage R² (passed): {passed_df['r_squared'].mean():.4f}")

# 4. Analyze failures
failed_df = flux_results_df[flux_results_df['qc_pass'] == False]
print(f"\nQC Failures:")
print(f"  By variable: {failed_df['variable'].value_counts().to_dict()}")
print(f"  By land use: {failed_df['land_use'].value_counts().to_dict()}")
print(f"  Reasons: {failed_df['note'].value_counts().to_dict()}")
```
</details>
{% endcapture %}

<div class="notice--primary">
{{ exercise | markdownify }}
</div>
</div>


### 4.6 Comparing Fluxes Across Land Use Types

The key scientific question: **Do forest and grassland ecosystems have different gas fluxes?**

### Creating Box Plot Comparisons

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Filter to QC-passed only
df_valid = flux_results_df[flux_results_df['qc_pass'] == True].copy()

# Get unique variables
variables = df_valid['variable'].unique()
n_vars = len(variables)

# Create subplots - one for each gas
fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 6))

# Handle case of single variable
if n_vars == 1:
    axes = [axes]

for ax, variable in zip(axes, variables):
    df_var = df_valid[df_valid['variable'] == variable]
    display_name = GAS_CONFIG[variable]['display_name']
    
    # Box plot showing distribution
    sns.boxplot(data=df_var, x='land_use', y='flux_umol_m2_s', 
                palette='viridis', ax=ax)
    
    # Overlay individual data points
    sns.stripplot(data=df_var, x='land_use', y='flux_umol_m2_s',
                  color='black', size=8, alpha=0.7, ax=ax)
    
    ax.set_title(f'{display_name} Flux by Land Use', fontsize=14)
    ax.set_xlabel('Land Use', fontsize=12)
    ax.set_ylabel('Flux (µmol m⁻² s⁻¹)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

### 4.7 Exporting Results

Finally, save your results for future use or reporting:

```python
# Export full results
flux_results_df.to_csv('flux_results.csv', index=False)
print("Saved: flux_results.csv")

# Export summary by gas and land use
summary = df_valid.groupby(['variable', 'land_use']).agg({
    'flux_umol_m2_s': ['count', 'mean', 'std', 'min', 'max']
}).round(6)
summary.columns = ['n', 'mean', 'std', 'min', 'max']
summary.to_csv('flux_summary.csv')
print("Saved: flux_summary.csv")

summary
```


