---
title: "2. Data Handling and Visualization"
layout: default
nav_order: 4
---
# **Of plots and pandas: Data handling and Visualization**

In this second part of the course we will talk about how to handle, process and visualize data in Python. For that purpose we will make use of a few third-party libraries. NumPy and Pandas will help us store the data in array- and matrix-like structures (in Pandas more specifically Dataframes) and do some processing of the data. Pandas already has some visualization capabilities, but for nicer looks and configurability we will make use of the Plotly package.    
To underline that these are essential tools in Python, let me once again pull out the Stackoverflow 2023 survey: According to the ~3k respondants, Numpy, Pandas and Scikit-Learn (which we will use in the next lesson) are 3 of the 8 most used technologies in programming across all languages (disregarding web-technologies)!  
  
![Stackoverflow 2023 survey technologies](assets/images/python/2/technologies.PNG)  
  
For this part we will use some example data. It is a dataset from the german weather service DWD from the Diepholz Station (ID 963) ranging from 1996 to 2023. [Click here to download (25mb)...](assets/data/dwd_diepholz_1996_2023.parquet).  
  
**Note** The data is in .parquet-format. You may not have heard of it, but this is a very compressed and fast format. For example this dataset with 27 years worth of data, in Parquet this is 25mb of data, in .csv its 208mb.  
While you can not open .parquet directly in excel or a text editor like a .csv file, it is much much faster to load e.g. when using it in programming languages, which is exactly what we are going to do here.
{:.notice}
  
As a last note: NumPy is one of the older Python libraries and Pandas is actually built on top of it. However, because we work with example data and want to get hands-on as fast as possible, we will cover Pandas first and then go from there.  


#### Table of Contents
1. [Pandas](#1-pandas)
2. [NumPy](#2-numpy)
3. [Plotly](#3-plotly-the-modern-plotting-library)

## 0. Importing modules
Just a quick forword on importing libraries in Python. Pandas, Plotly and Numpy are all external libraries we need to import to our script in order to make them work. Usually we would also have to install them, but since we work in Anaconda, this is already taken care of for us!  
Very simply, to import a library you type "import" and then the respective name. Typically you want to give an "alias" to the package, which is basically a variable that you can then use to access all the methods in the package. For some packages there are long-standing standards of what names to use. For pandas for example this is "pd":

```python
import pandas as pd
```

You can also only import specific parts of a package, which can save memory. Going back to one exercise from the previous lesson, if you know that you will only use the sqrt function from the math package you can use the syntax
```python
# Importing only a single function, squareroot
from math import sqrt

# Importing several functions, squareroot and greatest common divider
from math import sqrt, gcd

# theoretically you could also give an alias here
from math import sqrt as squareroot # this does not make much sense though
```

## 1. Pandas
Pandas is around since 2008 and one of the most wiedely used tools for data analysis of all. The usage is all about two types of objects: The pandas Series and the Pandas DataFrame, where a Series is more or less one column of a dataframe (basically a vector). If you already worked with R, the concept of a DataFrame is not new to you. However for starters, a DataFrame is basically a table, in which each row has an index and each column has a label. Simple right?  
  
![Pandas Dataframe Strcuture](assets/images/python/2/pandas_df.png)  
(credit: https://www.geeksforgeeks.org/creating-a-pandas-dataframe/)
  

### Creating DataFrames
Lets create a first little DataFrame. There are several ways to do it, one rather intuitive way is to use a dictionary. Think about it, a dictionary already has values which are labeled by keys. You can easily imagine this in a table-format: The keys will be the column-labels and the indices (row-labels) are by default just numbered.  
```python
import pandas as pd

# Note that we create an instance of the class "DataFrame"
# Therefore we have to call the function pd.DataFrame(). Within
# the brackets we then define a dictionary using the {}-style syntax
values_column_1 = [2,4,6,8,10]
values_column_2 = [3,6,9,12,15]
df = pd.DataFrame({
    "column_1": values_column_1,
    "column_2": values_column_2
})
```  
  
Another option to create a dataframe is of course to read in data. Lets go ahead and read the data from the german weather service that you can download above. Now we can use pandas built-in data-reader to directly create a DataFrame from the parquet-file:  
```python
# The path can either be the absolute path to the place where you saved the file
# or the relative path, meaning the path relative to the place where your script is.
# I'd recommend to create a subfolder where your script is called "data" and then
# import the data from the path "./data/diepholz_data_1996_2023.parquet"
df_dwd = pd.read_parquet('path_to_file')
```

### Accessing rows and column
Once you createad a dataframe, you can access individual columns by using the column names. Either you can directly access them using brackets, or you use the built-in ".loc"-function. I would recommend getting used to the .loc right away, as it rules out some errors you can run into otherwise. With .loc you always have to provide first the rows you want to access and then the column, separated with a comma. If you want to get all rows, that is done using a colon (":") To get a list of all availabel columns you can simply type "df.columns"  
  
```python
# First we can take a look at the available columns
df_dwd_columns = df_dwd.columns
print(df_dwd_columns)

# Then we can use the column names to extract a column
# from the dataframe
# Either you use only the column name in brackets:
df_dwd["tair_2m_mean"]
# But even better: use the .loc function:
df_dwd.loc[:,"tair_2m_mean"]        # get all rows
df_dwd.loc[20:50,"tair_2m_mean"]    # get rows 20 to 50
df_dwd.loc[:20,"tair_2m_mean"]      # get all rows up to 20 (including 20)
df_dwd.loc[20:,"tair_2m_mean"]      # get all rows after 20 (including 20)
```
Note that the .loc examples above all assume numeric index. But Pandas is not restricted to that!
The index (or "row-label") could also be something like "mean" or "standard-deviation".  
Keep that in mind for the exercise below!

### Built-in methods to describe the data
Pandas has a great set of convenience functions for us to look at and 
evaluate the data we have.  
- .info() gives us a summary of columns, number of non-null values and datatypes  
- .head() and .tail() show the first or last five rows of the dataframe  
- .describe() directly gives us some statistical measures (number of samples, mean, standard deviation, min, max and quantiles)  
Note that the output of .describe() is again a DataFrame, that you can save in a variable to evaluate it.  
There are also built-in methods that you can run directly on single columns. Examples of such functions are .mean(), .min(), .max() and .std().
  
{% capture exercise %}

<h3> Exercise </h3>
<p >You already know, how to call a method that is attached to a class. With that knowledge, explore the Diepholz DWD dataset and figure out the mean, standard deviation, min and max for 
air temperature, precipitation height, air pressure and short wave radiation (SWIN)</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Hint</summary>
It may be that the output of the .describe() function has a pretty bad formatting with 5 decimal numbers or more. In that case you can change the formatting of the output using  
```python 
df.describe().map('{:,.2f}'.format)
```
</details>

<details><summary markdown="span">Solution!</summary>

```python
# There are lots of ways to complete this exercise.
# You can use the above mentioned describe() method
# First get the summary. Save the output of .describe()
# in a new dataframe
df_dwd_summary = df_dwd.describe().map('{:,.2f}'.format)

# Then you can access values in that dataframe like this:
tair_2m_mean = df_dwd_summary.loc["mean", "tair_2m_mean"]
tair_2m_min = df_dwd_summary.loc["min", "tair_2m_mean"]
# and so on...

# You could also directly use the pandas built-in .min, .max,
# .mean and .std methods. For example:
tiar_2m_mean = df_dwd["tair_2m_mean"].mean()
tiar_2m_min = df_dwd["tair_2m_mean"].min()
# and so on...

```
</details>

<h3>Challenge </h3>
There is a one-line solution to this task, that only grabs the values asked for in the exercise. I wouldn't say that that would be the recommended solution for the sake of overview, but to fiddle around it is a good challenge. Hint: You can pass lists for the row- and column-labels to .loc
<details><summary markdown="span">Solution</summary> 

```python
# We can chain all the commands above to a one-line operation, meaning we 
# directly call .describe().map().loc[] on each others output.

# By passing the list ["mean", "std", "min", "max"] as row-indices and 
# ["tair_2m_mean","precipitation","SWIN","pressure_air"] as column-labels 
# we can directly access the range of values asked for in the exercise.

df.describe().map('{:,.2f}'.format).loc[["mean", "std", "min", "max"], ["tair_2m_mean","precipitation","SWIN","pressure_air"]]
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

### Datetime 
Pandas has a specific datatype that is extremely useful when we are working with time series data (a s our example DWD dataset). It is called datetime64[ns] and allows us to do a range of super useful things like slicing based on dates or resampling from 10-minute to daily, weekly, monthly data and so on. With datetime-indices, handling timeseries gets so much more convenient.  

```python
# get data newer than 31.12.2022
df_dwd[df_dwd["date_time"] > "2022-12-31"]

# get only data from 2022
df_dwd[df_dwd["date_time"].dt.year == 2022]

# But wait! Its not working, is it?
# Can you figure out why not? Remember the type() function!
```
Now, how do we get this to work for us? Well, the methods work with the datetime64 data type, so we need to change the "date_time" column type! Luckily, Pandas has a function for that. It is called to_datetime() and is part of the main library, so you call it as pd.to_datetime(). It takes the column you want to convert to datetime64 type as argument, tries to parse it to datetime64 and returns the result series. If it fails to parse, maybe because your date_time column is in a country-specific formatting, you can pass an additional "format" argument in which you provide the input format. But we will not cover it here, as the default should work for the DWD dataset.  
  
```python
example_df = pd.DataFrame({
  "date_time": ["2022-01-01 01:00:00","2022-01-01 12:00:00", "2022-01-02 01:00:00", "2022-01-02 12:00:00", "2022-01-03 01:00:00", "2022-01-03 12:00:00"],
  "values" : [1,5,4,20,6,-10]
})
type(example_df["date_time"])
example_df["date_time"] = pd.to_datetime(example_df["date_time"])
```

{% capture exercise %}

<h3> Exercise </h3>
<p >1. In your dataframe, turn the "date_time" column into a datetime64 type column. Then create dataframes for each season across all years, 
meaning one for spring, summer, autumn and winter each. The respective months are March to May, June to August, September to November and December to February. Compare the mean air temperature, precipitation and radiation
between the different seasons.
<br>
<br>
2. Find the dates of the maximum temperatures measured in the dwd dataset.
<br>
<br>
<b>One hint</b>: What we want to do here is to find those rows, where the value is one of a set of values.
To do so you can use the built-in pandas function .isin(). An example:</p>

{::options parse_block_html="true" /}

```python
# Here is an example series (representing a column of a dataframe)
series = pd.Series([1,2,3,1,2,3,1,2,3])
# Wen want to extract the rows where the value is 1 or 3:
desired_values = [1,3]
series_ones_and_threes = series[series.isin(desired_values)]
# Note that the indices in the extracted series are the ones from series, where the value is 1 or 3,
# so it really represents an extracted subset of the original series
```

<details><summary markdown="span">Solution!</summary>

```python
df_dwd["date_time"] = pd.to_datetime(df_dwd["date_time"])
df_dwd["date_time"] = pd.to_datetime(df_dwd["date_time"])

# First of all we create 4 dataframes, one for each season
# We do it by accessing the numeric value of the months in the "date_time"
# column. 1 refers to January and so on. With the .isin() method we extract
# those rows where the values correspond to the numbering of the month
df_dwd_summer = df_dwd.loc[df_dwd["date_time"].dt.month.isin([6,7,8])]
df_dwd_autumn = df_dwd.loc[df_dwd["date_time"].dt.month.isin([9,10,11])]
df_dwd_winter = df_dwd.loc[df_dwd["date_time"].dt.month.isin([12,1,2])]
df_dwd_spring = df_dwd.loc[df_dwd["date_time"].dt.month.isin([3,4,5])]

# To find the mean for each season we have a range of different options
# how we want to get the means and compare them. I'll show three different
# ways which are all valid.

# We know we will want to do some operation on all of the 4 datasets, so it is
# already a good idea to put them in a list. That way we can easily iterate over them
seasonal_datasets = [df_dwd_spring, df_dwd_summer, df_dwd_autumn, df_dwd_winter]

# Now one option would be to iterate over the datasets and print 
# the mean values of the desired columns:
seasons = ["spring", "summer", "autumn", "winter"]
for idx, df in enumerate(seasonal_datasets):
    print("----------")
    print(seasons[idx])
    print("----------")
    print(f'mean Ta: {df["tair_2m_mean"].mean()}')
    print(f'mean precipitation: {df["precipitation"].mean()}')
    print(f'mean SWIN: {df["SWIN"].mean()}')
# This way we have the outputs grouped by seasons

# Another option would be to iterate over the variables we want 
# to evaluate. Then we can print the variable values for each
# season directly below each other:
variables = ["tair_2m_mean", "precipitation", "SWIN"]

for idx, variable in enumerate(variables):
    print("--------")
    print(variable)
    print("--------")
    for i, df in enumerate(seasonal_datasets):
        stats = df.describe()
        print(f"{seasons[idx]}: {stats.loc['mean', variable]}")

# Often times we don't even want to print the output but rather
# just extract and keep it for later use, e.g. for visualizing it later.
# So another option is to create a new dataframe that holds
# the seasons as columns and variables as rows. That way we can 
# just look at the whole new dataframe and easily compare the values
seasonal_df = pd.DataFrame(columns = seasons)

for idx, df in enumerate([df_dwd_spring, df_dwd_summer, df_dwd_autumn, df_dwd_winter]):
    season = seasons[idx]
    seasonal_df.loc["Ta", season] = df["tair_2m_mean"].mean()
    seasonal_df.loc["Precip", season] = df["precipitation"].mean()
    seasonal_df.loc["SWIN", season] = df["SWIN"].mean()

print(seasonal_df)

```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
  
In this exercise we extracted seasonal information from 5-minute interval data. This type of frequency-conversion is something we do very often when working with time-series data. We also call this operation "resampling". Pandas actually has a great convenience function, that makes resampling a breeze, utilizing the wonderful datetime64-format.  
  
The operation consists basically only of two function calls on the pandas dataframe. The first is ".resample()". We must define the column that contains the datetimes with the "on" argument and our target frequency with the "rule" argument as a string. The most useful frequency specifiers are:
- "S": seconds
- "T" or "min": minutes
- "H": hours
- "D": days
All of these can be extended with numbers, such as "7D" for 7 days or "30min" for half-hourly values.  

Afterwards we also have to call a function that specifies **how** we want to resample. You see, if we change the frequency from 5 minute data to daily data, the daily value can be computed in different ways. For example for temperature it would make sense to use the daily mean value. For precipitation on the other hand it is probably more useful to get the daily sum, if we are interested in the amount of rain per day. That is why ".resample()" has to be followed by a function like ".mean()" or ".sum()". Here is a full example:  

```python
example_date_time = pd.to_datetime(["2022-01-01 01:00:00","2022-01-01 12:00:00", "2022-01-02 01:00:00", "2022-01-02 12:00:00", "2022-01-03 01:00:00", "2022-01-03 12:00:00"])

example_df = pd.DataFrame({
  "date_time": example_date_time,
  "values1" : [1,5,4,20,6,-10],
  "values2" : [100,500,400,2000,600,-1000],
})
df_daily_means = example_df.resample(rule="1D", on="date_time").mean()
df_daily_sums = example_df.resample(rule="1D", on="date_time").sum()
```

I mentioned before that pandas itself already has some built-in plotting capabilities. I won't go deep onto it, but it is definitely worth mentioning because you can use it to get a quick overview of data in a pandas dataframe. You can simply call the ".plot()" function on any dataframe. You can run this on the whole dataframe, which will plot all available columns, or you extract specific rows/columns with the methods we discussed before and then run ".plot()":  

```python
# lets use the example_df from above:
# First we plot both values1 and values2:
example_df.plot(x="date_time")
# Note that we have to say that we want date_time on the 
# x-axis, because otherwise it will by default use a numeric
# index on the x-axis and plot the date_time column against it 
# as well. You can try it out by leaving the x=... out
# The figure should now pop up in the "plots" tab on the right
# side of your Spyder window

# Now we just plot values 1:
exampl_df.plot(x="date_time", y="values1")
```
  
There is a lot more functionality, but I want to leave the pandas plotting with that, as we will dive deeper into plotting later in this course.  


Ok, we have covered quite some ground on handling pandas dataframes. We covered  
- how to create dataframes
- how to read data from .csv or .parquet files
- how indexing works
- how we get some descriptive information on the data
- how to compute some informative values such as the min, max and mean of a series
- even how datetime-indices work (honestly we just scratched the surface, but for an introduction course this is already quite advanced)
- and how to resample time-series data to another frequency
Finally I just want to give some "honorable mentions", to tell you about functions with pandas that you will probably need at some point.
No exercise here, I just want you to have heard of these:

```python
# 1. pd.concat():
# this function concatenates dataframes with matching columns or pandas series
# meaning it simply glues one dataframe on the bottom of the next:
df_1 = pd.Series([1,2,3])
df_2 = pd.Series([4,5,6])
df_3 = pd.concat([df_1, df_2]) 
# note that we have to put the two dataframes in a list

# 2. pd.merge()
# This functions combines dataframes based on common indices.
# It is a rather complex function but this is a simple example 
# how to combine two dataframes that have overlapping indices:
df_1 = pd.DataFrame(
    index = [1,2,3],
    data = {
  "col_1": [1,2,3],
  "col_2": [4,5,6]
  })
df_2 = pd.DataFrame(
    index = [3,4,5],
    data = {
  "col_1": [7,8,9],
  "col_2": [10,11,12]
  })
df_3 = df_1.merge(right=df_2, how = "outer")

# 3. df.apply()
# In the call to apply you can define a function that will be
# executed on each element of the dataframe:
df_1_plus_one = df_1.apply(lambda x: x+1)

# don't worry about the "lambda", it simply creates
# the variable "x" we can use for "x+1". x is only there
# during the computation and then immediately vanishes again
```

## 2. A quick touch on Numpy
Many Python programmers and data scientists would probably shun me for not giving more time to numpy,
but we want to get to the applications as fast as possible. However, if you want to know more you can 
- [download a little Numpy cheat sheet here](assets/cheatsheets/Numpy_Cheat_Sheet.pdf)
- [check out the official Numpy documentation](https://numpy.org/doc/stable/index.html){:target="_blank"}{:rel="noopener noreferrer"}
- [read about numpy at w3schools.com](https://www.w3schools.com/python/numpy/numpy_intro.asp){:target="_blank"}{:rel="noopener noreferrer"}  
Numpy is like the grandmaster of handling data in Python. It has always been there, it can do everything, but it is not neccessarily pleasant to deal with.  
With Numpy you can create vectors and multi-dimensional matrices, do computations and much more. It is very lightweight (meaning it uses very little memory) and super fast.  
Actually, Pandas has Numpy as its underlying framework. Every column or row in a pandas datframe
is actually a Numpy array with fancy extras. That makes Pandas slower than Numpy but also much more convenient to use.  
While we can do most of our analysis in this course only with Pandas, I think you should know about the basic
functionality and the core uses of Numpy. So lets take a look at some simple structures and computations:
  
### 2.1. Numpy Arrays
The most used structure in Numpy are arrays. In contrast to normal Python lists, they are faster, they force the values to be homogeneous (e.g. no strings and integers mixed in a Numpy array), 
and with Numpy arrays you can compute some mathematical operations between arrays such as element-wise addtion, cross-products and so on. 
Additionally, numpy provides a range of functions you can run directly on arrays, such as .mean(), .min(), .max(), .median() and so on.  
Generally you can think of Numpy arrays/matrices vs Pandas Dataframes/Series as the difference between pure vectors or structures
with pure numeric data in them vs. fully fledged and labeled tables.  

There are different ways to create Numpy arrays: 
```python
import numpy as np
# a simple vector is created by calling np.array 
# with a list as argument:
vector_1 = np.array([0,1,2,3,4,5])

# alternatively, you can directly create a vector
# filled with zeros or ones providing a shape.
# The shape has round brackets and defines the 
# dimensions of the data structure. For example
# (2,3) will create a matrix with 2 rows and 3 columns
vector_zeros = np.zeros(shape=(2,3))
vector_ones = np.ones(shape=(2,3))

# with np.random.rand() you can create a matrix with 
# random elements between 0 and 1, by multiplying it
# you can get e.g. values between 0 and 10:
vector_randoms_0_to_1 = np.random.rand(3,10)
vector_randoms_0_to_10 = np.random.rand(3,10)*10

# You can then get individual elements from that 2-D
# structure with indexing. For example to get the
# the second element in the first column:
vector_randoms_0_to_10[0,2] = 2 

# You can find the shape of a numpy object with
vector_zeros.shape()

# Lastly you can create arrays with consecutive numbers with np.arange()
# I takes a start, an end and an interval as arguments:
range_10 = np.arange(0,10,1)
range_10_halfsteps = np.arange(0,10,0.5)
# The ranges are created including the first and excluding the 
# last number.
```

### 2.2. Useful Numpy functions
In addition to Numpys own data structures it provides a whole range of 
useful functions that can be used in other contexts as well.  
One function I probably use more than any other are np.floor(), np.ceil()
and np.round(). These all round values. Floor returns the nearest lower integer,
ceil the nearest upper integer and round rounds to a desired decimal point:

```python
vector = np.array([1.1, 10.523124, 3.341])
vector_ceiled = np.ceil(vector)
vector_floored = np.floor(vector)
vector_rounded = np.round(vector, 2)
```

Numpy also provides some mathematical functions and constants. For example  
np.pi returns the value of pi, np.e returns Eulers number.  
Other mathematical operations include all angle computations such as np.sin(), np.cos() etc.
These are all computed in radians, but you can turn them into degrees with np.degrees()

{% capture exercise %}

<h3> Exercise </h3>
<p >Lets just do one quick exercise on numpy to get familiar. <br>
1. Create a numpy array from 0 to 20 in steps of 0.1. <br> 
2. Compute the sin of the data, then compute the standard deviation of the sin data <br>
3. Add some random noise to the data. To do so, use the np.random.rand(). The range of the noise
should be between 0 and 0.5. Then compute the standard deviation of the noisy data. <br>
4. Round the noisy values to 3 decimal places <br>
 </p>
{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
vector = np.arange(0,21,.1)
sin_vector = np.sin(vector)
std_sin_vector = sin_vector.std()
noisy_sin_vector = sin_vector + np.random.rand(len(sin_vector))*0.5
std_noisy_sin_vector = noisy_sin_vector.std()
rounded_noisy_sin_vector = np.round(noisy_sin_vector, 3)
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>


## 3. Data Visualization: Plotly
Finally! It is time to not only create endless boring arrays of numbers, but to mold them into beautiful, descriptive images that tell the story of what the data actually means. Because that is essentially what we are doing when plotting data. Nobody can look at a table of 100.000 rows and start talking about it, that is what we can achieve with data visualization.  
![plots of amount of black ink](assets/images/python/2/self_description.png)  
  
There are several libraries we could use for plotting in Python:
- Matplotlib: One of the most widely used frameworks. It is lightweight, built into Pandas but nobody really likes the syntax
- Seaborn: A library built on top of Matplotlib. It makes the syntax quite a bit easier, provides nice out-of-the-box plot styles, but the documentation is a bit lacking and plots are not that easy to customize
- **Plotly**: The solution we will be using here. Plotly is built on a Javascript library Plotly.js and therefor brings some unique features to the table. The syntax and strcuture is quite good to learn, it offers a load of customization. Additionally, it offers very nice interactivity with the plots which makes exploration of your data much easier

Lucky for us, Plotly is already included in Anaconda, so we do not need to install it.  
Plotly provides two different approaches to plotting:  
- Quick and easy plots with less customization using plotly express
- fully fledged figures with full customization options using graphic-objects

To get a good understanding of Plotly it makes sense to go from large to small, first looking at the general structure of Plotly figures and the way graphic_objects work. If you have a broad overview of these you can still learn about the quick-and-easy ways, but you will have a much easier time when you want to change something about the express solutions manually.  
  
### Plotly - The modern plotting library

#### Where to find help

First of all lets gather some ressources. The two best places to find advice about any plotly-related questions are 
- [the official documentation at plotly.com](https://plotly.com/python/){:target="_blank"}{:rel="noopener noreferrer"}
- [the plotly community forum](https://community.plotly.com/){:target="_blank"}{:rel="noopener noreferrer"}
- as always, Stackoverflow...

#### The general structure of Plotly figures
First of all we need to go through a little bit of vocabulary to be able to talk about Plotly. In Plotly-world the whole image of a plot, including the axes, the data, the labels, the title and everything is called the "graph-object". This is the top-level of every Plotly figure and it is also the name of the Python class, with which we build the plots.
Within the graph-object there are two layers:  
One is the "data" layer with everything that is directly related to the displayed data. That is the data itself, the mode of repesentation in the graph for example the line (in a line-plot) or points (in a scatter-plot) and the styling such as the size or color of the line/points. In plotly, they also call the group of data-related attributes "traces". Don't ask me how they came up with it but we have to live with it... We will come back to that later!  
The second part of the figure is the "layout" layer. It includes everything that makes the graph besides the data itself, for example the axes, the titles on the axes, the title of the graph itself, the legend, colors, maybe a template and so on.  
In the image below I tried to highlight the areas including the "data" area in red and the "layout" related areas in green:    
![Plot with marked data and layout areas](assets/images/python/2/ta_2m.png)  

Lets dive into the code and create a first figure object. Its easy:  
```python

# Before we start, lets resample the dwd data down to daily values.
# You will create quite some plots and plotting 27 years of 10-minute
# data takes a bit of time.
df_dwd_daily = df_dwd.resample("d", on="date_time").mean()
# First import the graph_objects module from plotly.
# We call it "go" because that is convention 
import plotly.graph_objects as go
# Then we create out figure like this:
fig = go.Figure()

# Check out what happens, when you print this 
# object with print(fig). You will see the structure 
# we talked about above!
```
Well, now we have a graph-object without any data. From printing the figure you can see that the "data" is an empty list.  
Lets change that and add some data from out dwd-dataset. To do so, we have to add a "trace" (remember how we introduced that above). We do that by calling the .add_trace() method on the figure.  
In the function the first thing we have to define is, what kind of graph we want to create. Otherwise the empty figure wouldn't know whether it should become a scatter-plot, a histogram, a line-plot or anything else. We define the type of graph by giving an object of the graph-type we want to the "add_trace()" method. These objects are also included in our "graph-objects" (or "go"). Sounds complicated, but really it is not. Check this out:  

```python
# This is the bare figure
fig = go.Figure()
# Now we will add some data:
fig.add_trace(  # On fig we call the "add_trace()" method
  go.Scatter(   # In the method we provide an object of type "Scatter" from "go"
    x = df_dwd_daily.index,  # then, in go.Scatter we define, which data should be plotted
    y = df_dwd_daily["tair_2m_mean"],# on the x- and on the y-axis
  )
)
# Now print the figure again and look the output
# You will see that the "data" level now has the x- and y-data in it
# Plotly has very nice interactivity. To open the graph
# in an interactive browser-window type this:
fig.show()
# or you save the figure to an image like this:
fig.write_image("daily_tair_2m_mean.png")
```
Above we created a scatter-plot (every data point is a dot in the graph). But if you look at the plot, you'll note that there is still a lot missing. Most importantly, it does not have axis-labels. We need to add those, so people know what is plotted here! Lets do it. Which part of the figure do you think we need to change to add axis-labels?  
<details><summary>Solution</summary>The "layout" bit of the figure </details>  
  
So lets see how we can change the layout of the figure!  
```python
# to get to the layout of the figure we have two options:
# 1. The figure object "fig" has the "layout" property,
# which has an "xaxis" property, which again has a "title"
# property. We can go down this path manually like this:
fig.layout.xaxis.title = "Date"
fig.layout.yaxis.title = "Tair [F]"

# 2. The second option is to use the "update_layout() method.
# This was was made to make styling more convenient. We can use 
# it to "group" our styling in a single function call. 
fig.update_layout(
  xaxis_title="Date",     # Note that we use an underscore
  yaxis_title="Tair [F]"  # "_" to grab the "title" property from "xaxis"
)

# Now you'll see, that the labels are changed in the figure:
fig.show()
```
This is pretty much the way you can change any attribute that is related to the layout of the figure. The only thing you have to figure out for whatever you want to change in your figure is, where the respective property lies. Is it part of the data or the layout layer? Which sub-layers are there? Sometimes you can figure it out by thinking about it, however you can always refer to the documentation and the hive-mind of the internet. Especially in the beginning you'll need to google quite a bite, but once you get the hang of it, it is actually quite intuitive.  
Lets do some more styling. Above we created a scatter-plot. This is a time-series, so maybe a line-plot would be more appropriate...

{% capture exercise %}

<h3> Exercise 1 </h3>
<p>Try to change the style of our plot above to a line-plot. To do so, you need to change the "mode" property which is part of the "data" layer, or "trace". You can change the trace just like we changed the "layout" above with a function called "update_traces(). <br>
<b>Challenge:</b> Can you come up with two different ways to change the mode? 
</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
# Option 1:
fig.update_traces(
  mode="lines"
)
# Option 2 (which you usually wouldn't use):
fig.data[0].mode = "lines"
# The trick is that we have to write 
# fig.data[0], because the "data" property is
# a list! You can see that if you look at the 
# printed figures "data" property, it starts with a "[".
# The reason is of course that you could plot several
# lines within a single plot. This way you could style
# them one-by-one. However, generally you use the 
# "update_traces()" method to apply styles that 
# are used for all plotted data and pass everything
# else directly when you create the data with "add_trace()"

```
</details>

{::options parse_block_html="false" /}

<h3> Exercise 2 </h3>
<p>Now lets expand the plot a bit. Add two more lines to the plot, the tair_2m_min and tair_2m_max columns from our dwd data. You can simply add them to the existing plot with the "add_trace()" method. When calling add_trace(), try to directly change the mode to "lines". <br>
When adding the lines, also add the argument "name" to the add_trace() method. That defines, how the line will be reprented in the legend. Give appropriate names to the lines. <br>
Additionally, try to change the line style of the min and max temperature to "dashed". If you want, you can also change the colors of the lines. To do so, change the line_color property. To define the color you can use either a string in the form of "rgb(0,0,0)" where you have to replace the zeros with rgb values, or you use one of the pre-defined colors which you can also pass as string. You can find a list of available color-names here:<br>
</p>
 <a href="https://www.w3schools.com/cssref/css_colors.php">w3schools list of CSS colors...</a>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
fig = go.Figure()
fig.add_trace(
  go.Scatter(
        y = df_dwd_daily["tair_2m_mean"],
        name="Tair 2m",
        line_color="black"
    )
)
# after adding the first line we just keep adding
# more lines. We can directly change the name,
# line_dash and line_color attributes:
fig.add_trace(
  go.Scatter(
        y = df_dwd_daily["tair_2m_min"],
        name="Tair 2m min",
        line_dash="dash",
        line_color = "lightblue"
    )
)
fig.add_trace(
  go.Scatter(
        y = df_dwd_daily["tair_2m_max"],
        name="Tair 2m max",
        line_dash="dash",
        line_color="lightcoral"
    )
)
fig.update_layout(xaxis_title="Date", yaxis_title="T2m [F]")
fig.show()
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

Great, you are on the best way to becoming a Data-Painting Plotly-Wizard!  
Of course there are not just simple line and scatter charts. [There is a whole world of graphs to explore!](https://plotly.com/python/){:target="_blank"}{:rel="noopener noreferrer"}. For now, lets look at just one more type of graph, a bar-chart. This is a common type of graph to compare measured amounts (as opposed to discrete values such as a temperature). Such a value would be our rainfall measurement!  

{% capture exercise %}

<h3> Exercise </h3>
<p >Go ahead and create a bar chart of the sum of daily rainfall. You can create a bar-chart just like we did with the scatter plots above, only that you call "go.Bar" instead of "go.Scatter" <br>
<b>Remember</b> that when resampling precipitation to daily values, you need to use the sum instead of the mean!</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
# First of all we grab daily precipitation data by 
# resampling with the "sum" aggregation function
# Here I directly grabbed only the precipitation-column, 
# but you can also do it another way
precip = df_dwd.resample(rule="d", on="date_time").sum()["precipitation"]

# Now we create the graph just like before:
fig_precip = go.Figure()
fig_precip.add_trace(
    go.Bar(           # Here we simply use go.Bar instead of go.Scatter
        x=precip.index,
        y=precip      # Note: when using a Series instead of a dataframe
    )                 # I dont have to pass the column name, because I 
)                     # only have one column anyways...
fig_precip.update_layout(
    xaxis_title="Date",
    yaxis_title="Rain amount, daily [mm]"
)
fig_precip.show()
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>

Right on, this was quite a deep dive into the Plotly library! But if you followed all the way down here, you are on a very good way to become super proficient in plotting data in python! The skills you got from the exercise above should get you quite far in designing your own figures in the future.  
If it is all a bit much in the start, don't worry! As time comes you will do things much faster. For now, keep trying, keep googling, consult the documentation and most importantly: be happy with the progress you make!  
  
Before we finish the visualization exercises I want to show a few more very helpful things.  

#### Subplots
Often you want to create not just one but multiple plots in one figure, for example one big figure with a temperature plot on top and a precipitation plot on the bottom. This way readers can easily get an overview of the climate at the station.  
Creating such a "subplot" in Plotly is super easy! Instead of using go.Figure(), you use a different function to create your top-layer "graph-object". The function we need is plotly.subplots.make_subplots(). In it we can define the number of rows and columns of figures we want to create with the "rows" and "cols" keywords. Think about the whole figure like a matrix. The figure on the top-left will be row 1, column 1, second on the left row 2, column 1 etc.  
Then whenever you are adding a new trace, you can define its position with the properties "row" and "col":
```python

# First we create the subplots graph_object.
# To do so we have to import that specific method:
import plotly.subplots.make_subplots
# Now we create a subplot figure with two rows:
fig_subplots = plotly.subplots.make_subplots(rows=2, cols=1)
# Now we can start adding traces to the figure:
fig_subplots.add_trace(
    go.Scatter(
        x=df_dwd_daily.index,
        y=df_dwd_daily.tair_2m_mean,
        name="Tair mean",
        line_color = "black"
    ),
    row = 1,  # here we define, where the figure should be
    col=1,
)
fig_subplots.add_trace(
    go.Bar(
        x=precip.index,
        y=precip,
        name="precip",
        marker_color="blue"
    ),
    row = 2,  # precipitation will be the lower plot
    col=1
)

fig_subplots.show()
```

As a little exercise, print the fig_subplots object from above and try to figure out how to change the y-axis titles on the first and the second plot.
<details><summary>Solution</summary>
fig_subplots.update_layout(
    xaxis2_title="Date",
    yaxis_title="Tair 2m [F]",
    yaxis2_title="Rain amount, daily [mm]"
)
</details>

#### Templates
One very nice way to style your figure a bit differently than the default is to use Plotly templates! You can implement them in your figure simply by adding the template in the layout:  
```python
fig_subplots.update_layout(
  template="simple-white"
)
fig_subplots.show()
```  
Looks nice right?  
There is a whole gallery of templates available on the website:  
[Plotly template gallery...](https://plotly.com/python/templates/){:target="_blank"}{:rel="noopener noreferrer"}

#### Plotly Express
For now, we will leave it with that. But wait, I was talking about an easier way to create graphs before right?  
Yes, for "quick and dirty" graphs you can use the awesome plotly.express shortcut. With it you can create a bunch of graphs like the ones  we talked about above without all the fuzz of graph_objects etc.  
All you need to do to create a scatter plot is  

```python
import plotly.express as px
fig_express = px.scatter(df_dwd_daily, y="tair_2m_mean")
fig_express.show()
# You can update the fig_express just the same as the output
# of go.Figure(), with update_layout and all of its beauty.
```  

Plotly express is very well connected with Pandas. This enables you to display all data of a dataframe in one combined, interactive overview char, simply by passing a pandas dataframe into it:  

```python
import plotly.express as px
fig_express = px.scatter(df_dwd_daily)
fig_express.show()
```   

Plotly express includes many other graph types as well. You can find a very nice documentation [on the plotly website...](https://plotly.com/python/plotly-express/){:target="_blank"}{:rel="noopener noreferrer"}.   


One very last very useful thing I want to mention here is the addition of trendlines in Plotly Express. It is a super handy feature that is only implemented in Plotly Express, not in plain Plotly. I will show you how to do it and how you can grab all information the trendline can give you, but I will not explain the statistics behind it here.  
It happens very often that you want to plot two variables together, to see whether they are related. In order to explore the relationship, you can fit a linear line to the data and look at the parameters of the line-equation. For example, if I would plot air temperature on the x-axis against the same air-temperature on the y-axis, that would result in a perfect fit, the line would have a slope of exactly 1, the y-axis-intercept would be 0 and the r_square value of the regression would be 1, indicating the perfect fit.  
In Plotly express you can simply add a trendline to the plot by adding the "trendline" argument to the function call:

```python
import plotly.express as px
fig_express = px.scatter(df_dwd_daily, x="pressure_air", y="tair_2m_mean", trendline="ols")
results = px.get_trendline_results()
print(results)
fig_express.show()
``` 
But the bare line itself is not too useful, we need to grab the coefficients of the line, so the slope and the intercept of the line-equation.  
You can grab the results from the figure using px.get_trendline_results(fig_express). Honestly, the actual parameters are a bit hidden inside the deeper objects, but you can extract each one anyways:


```python
import plotly.express as px
fig_express = px.scatter(df_dwd_daily, x="pressure_air", y="tair_2m_mean", trendline="ols")
results = px.get_trendline_results(fig_express)

# To actually access the results of the regression we need to 
# access the first row of the px_fit_results:
ols_results = results.px_fit_results.iloc[0]
print(ols_results.summary())

# ols_results is an object of the type "OLSResults"
# which comes from a different library, "statsmodels" (https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.html)
# From it you can grab the coefficients and a bunch of statistical infos:

slope = ols_results.params[1]
intercept = ols_results.params[0]
rsquared = ols_results.rsquared
# etc...
``` 

I won't go much more into the details because you can easily look up more plotly express functions (and I do encourage you to do so because it is super handy!), but by now you know enough about Plotly to explore that yourself.  
Why did we go through all the fuzz of handling the traces and layout ourselves? Because express is limited! If you want to customize your plots in some way it does not support out-of-the-box, you **will** have to dive into the figure-structure sooner or later, and now you know how.  
Still it is encouraged (even by Plotly themselves) to make use of both: run plotly express for a base figure or for quick data exploration, and then style the figure the way you want with the in-depth methods.  

## Congratulations
on finishing this plot about plotting plots with Plotly! What a journey. I hope you can take away the knowledge to navigate plotting in Python fairly well from now on! Enjoy your future data-adventures and I wish you happy coding!  
![Clapping](assets/images/python/2/congrats_2.gif)
