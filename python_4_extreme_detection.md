---
title: "4. EXTREME VALUE DETECTION"
layout: default
nav_order: 6
---

# **Extreme value detection**

In this exercise we will look at extreme values in meteorological data. First you will learn about different ways to define what is an "extreme" 
value (extreme relative to what?). Afterwards we will work with an example dataset and code the methods in Python to get some hands on experience.

### Table of Contents
1. [Material](#1-material)  
2. [Background](#2-background)  
3. [Methods and Implementation](#3-methods-and-implementation)  
  3.1. [Peak Over Threshold (POT)](#31-peak-over-threshold-pot)  
  3.2. [Block Maxima (BM)](#32-block-maxima-method-bm)  
  3.3. [Moving Average (MA)](#33-moving-average-method-ma)  


## 1. Material
We will once again use the DWD dataset from the Diepholz station for this section. Below you can find the download again:
[Diepholz DWD meteo data (25mb)...](assets/data/dwd_diepholz_1996_2023.parquet).  
[Literatur](/assets/r_ex4/wmo-td_1500_en.pdf)  

We will have to do some plotting again, so it might be good for you to resample the data to daily data, just to reduce the size
of the dataset a bit.
By now you should know how to do it. Try to create a pandas Series with hourly air temperature data.  
If you are getting stuck, you can refer to the exercise before.

## 2. Background
We will look at three different methods to determine extreme events from time series of meteorological data. The main difference between the methods is the way they define the reference, to which we compare a value to describe it as being "extreme" or not.  
Pause for a second and think about how you could describe what an extreme value is.  
  
There are several ways to think about extreme values. An extreme value can simply be the highest/lowest value in a finite set of data. Think for example about testing the highest speed that a car reaches on a test drive. Here the absolute peak value would be a reasonable value of interest.  
  
In meteorological time series we are often interested in a range of extreme values. In other words, we are interested in the values in the tails of the distribution of our sample data, which exceed a certain threshold. It is important to consider the distribution of our underlying dataset and the question we actually want to answer. 

Our example dataset comprises of air temperature data from 1996 to 2023. If we are interested in the extreme values with respect to this whole time period, we can simply look at the distribution of all the data, determine a threshold and see which datapoints are above the upper or below the lower threshold. 

However, we might also be interested in the months with extreme temperatures. Because our distribution includes winter and summer data, extreme temperatures in spring and autumn will probably not be considered in this approach. For these we would have to create data distributions of seasonal, monthly or even daily data to evaluate extreme events on the respective time scale. This will become more obvious when we look at the methods.

<details>
<summary>
Read More: Extreme value return periods
</summary>
Another approach is the evaluation of extreme values and their probabilities based on historical data. Relating these probabilities to the time series of the data produces "return periods", frequencies in which the extreme values are expected to occur. As an example, requirements for buildings often include a resistance to weather extremes with a certain return period. Making up a case, wind turbines would be built that they can withstand windspeeds with a "return level" in a "return period" of 1 in 10.000, meaning the chance that such a windspeed occurs in a year would be 0.01%.
</details>

## 3. Methods and Implementation
Lets now look at three different methods to analyze extreme events in our sample dataset. We will talk about the reasoning and the implementations of the methods. 
We will then go through each method and implement the methods into functions, which you can then use to analyze your data.

As a little preface we need to talk about a concept we will use for all methods: Quantiles.  
What are quantiles?  
  
A quantile is a subset of the given data, that contains a certain percentage of the distribution of the data. E.g. in the following figure, everything left of the red line is in the q10 percentile, the lowest 10% of the data. Everything up to the blue line is in the q90, the quantity that comprises of 90% of the data.
![Quantiles](.\misc\2023\01\05\quantiles.png)
You can use the following function to visualize the quantiles of our dataset:
```python

def visualize_quantiles(x:pd.Series, q_low:float, q_high:float):
    import scipy.stats as stats
    x_mean = x.mean()
    x_sd = x.std()
    y = stats.norm.pdf(x.sort_values(), x_mean, x_sd) # This function creates the y-values of the normal distribution given our data, the mean and the standard deviation
    qh = x.quantile(q_high) # here we calculate the higher quantile threshold
    ql = x.quantile(q_low) # here we calculate the lower quantile threshold 
    fig = px.scatter(x=x.sort_values(),y=y)
    fig.add_trace(
        go.Scatter(
            x=[ql, ql],
            y = [0,max(y)],
            mode="lines",
            name=f"{q_low*100}% quantile"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[qh, qh],
            y = [0,max(y)],
            mode="lines",
            name=f"{q_high*100}% quantile"
        )
    )
    fig.show()

get_quantiles(df_dwd_ta_hourly, 0.05, 0.95)
```

Alright, now that we layed out the basics, lets dive into theme methods!  

![extreme Meme](assets/images/python/4/extreme_meme.png)  

### 3.1 Peak Over Threshold (POT)

The first approach is the Point Over Threshold (POT) method. This is a very simple approach that looks at the whole dataset as one.  
We define fixed thresholds for the dataset, defining the upper and lower bounds above or below which values will be considered extreme.  
The boundaries are usually defined by the quantiles we provided as an argument to the function.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Lets try and code that method ourselves. It is actually not very difficult! <br>
Define a new function called "peak_over_threshold()". <br> 
It needs to take a series of data as input and the quantile we want to use for extreme detection. <br>
Then we need to do the following operations: </p>
<ol>
  <li>Find the upper and lower thresholds for what is to be defined as extreme, absed on the quantiles. To find these values you can use the handy Python function "quantiles()". Just call it on the input Series and provide the quantiles as argument as in "X.quantiles(0.95)". Remember: You want the upper **and lower** thresholds. Think about how you can get both. </li>
  <li> Find those rows in the input series which are higher and lower than the upper and lower thresholds. You can get Series of booleans by comparing a pandas Series with a value. You can try it out, just type for example "X > 270" if X is your Series. </li>  
  <li> Finally you want to create a dataframe, because of course you want to return the results of your extreme value detection. Create a dataframe with the input data and two new columns, one containing the booleans of your high extreme values and the other for the low extremes. </li>
</ol>

A little hint: The description here is quite long but the code for this is actually quite short.  
  
{::options parse_block_html="true" /}

<details><summary markdown="span">Hint if you get stuck!</summary>
You can generate a Series of boolean values that indicate whether a datapoint is above or below a value with a direct comparison such as 
```python
X_larger_than_280 = X > 280
```
</details>

<details><summary markdown="span">Solution!</summary>

```python

def peak_over_threshold(X:pd.Series, prob):

    print(f'Extremes detection using peak over threshold method at: {prob} percentile')
    df = pd.DataFrame(index=X.index, data = {
        "data": X,
        "extreme_low":  X < X.quantile(q=1-prob),
        "extreme_high":  X > X.quantile(q=prob)
    })
    return df
```

</details>

For this and the next methods it will be very handy to have a function that plots the data and the extreme highs and lows in separate colors. You can try to build a nice plotly figure yourself or you use the code I provide below.

<details><summary markdown="span">Plot function</summary>

```python
def plot_extremes(data:pd.DataFrame, extr_high_col:str, extr_low_col:str):
    extr_high_data = data.loc[data[extr_high_col]==True, "data"]
    extr_low_data = data.loc[data[extr_low_col]==True, "data"]
    
    fig = go.Figure()
    fig.add_traces(    
        go.Scatter(
            x=data.index, 
            y=data["data"],
            mode="markers",
            name="no extreme",
            marker_color="black",
            marker_size=5,
            )
    ),
    fig.add_traces(
        go.Scatter(
            x = extr_high_data.index,
            y = extr_high_data,
            name="extr. high",
            mode="markers",
            marker_color='orange',
            marker_size=5,
            showlegend=True
        )
    )
    fig.add_traces(
        go.Scatter(
            x = extr_low_data.index,
            y = extr_low_data,
            name="extr. low",
            mode="markers",
            marker_color='LightSkyBlue',
            marker_size=5,
            showlegend=True
        )
    )
    fig.update_layout(
        template="simple_white"
    )
    fig.show()
    
```
</details>
  
{::options parse_block_html="false" /}

Take a look at the output and the datapoints marked as extreme values. 
Evaluate the plot yourself. What is the reference for these extreme values? 
Which questions could you answer with this type of extreme detection, which not?  


<h3> Exercise </h3>
<p>Lets fiddle with the code for a bit. Change the <b>prob</b> parameter to 85, 75 and see how the output changes. How many extreme values do you expect when setting prob to 50? Think about it and then run the function with that quantile. <br>

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>
### 3.2. Block Maxima Method (BM)

The next method we are looking at is the "Block Maxima" method. As the name states, we are looking at a certain "block" of data and find the maxima based on the defined threshold of the values in this block. There are several ways we could define these reference blocks. For example we could look at every year individually and find the extreme values for these. Alternatively, we could create blocks from each week of the year across all years or for every month across all years. We could then find extremes based on the quantiles of the data for every wekk of the year and separate e.g. extreme values in spring and autumn from the overshadowing extreme values in winter and summer.

In our example we will define the blocks as the values for each single day across all the years. The procedure is as follows: 

**Step 1**  
In the first line we again create a dataframe with the data, a date column and then add a new column called "DOY" with a mutation, that gives each date a value of 1 to 365. We need this "day" value to group our data across the years by it. We can get this by grabbing the "day_of_year" property from our datetime-indices in Pandas.

```python
df_bm = pd.DataFrame(index=X.index, data={
        "doy":X.index.day_of_year,
        "data":X.values,
    })
```

**Step 2**  
Next we create the column "data_14d_ma". This is the 15 day moving average around every day. Moving average means that the "window" of data we are calculating the mean from varies.

```python
df_bm["data_14d_ma"] = X.rolling(window=14, min_periods=1, center=True).mean()
```  

Through this, we accquire a smoothing of the daily temperature values and make the underlying dataset for our daily temperature distribution more broad. The reasoning is the following:  
We want to create a representative dataset for daily temperature values across the years. If we use the single day for each year, we have a dataset of 18 datapoints which can easily include heavy outliers. By using a moving average of 15 days we enhance our dataset for each day by a factor of 15 to 270 datapoints, still restricted to a pretty small time window. While it does reduce the impact of individual extremely hot or cold days, it is more likely to representatively capture the state of the atmosphere around the time of interest.  
  
**Step 3**  
Now that we have the smoothed data and our doy information we can go ahead and calculate the long-term mean for every day of the year. To do so, we use the pandas "groupby" function. This allows us to sample data based on common values in a column. E.g. for the day of the year, it will grab all values where the day of the year is 1 and calculate the mean for those, then for day 2 and so on.  

```python
long_term_means = df_bm.groupby("doy")["data_14d_ma"].mean()
```  
Now that we have those long term means, we can calculate the difference between every datapoint and the long-term mean that fits to its day of the year. To make it more clear, we can use the pandas "iterrows" function that allows us to loop through the rows of the dataframe.
First we create a new column filled with zeros called "diff". Then we go through the rows of the dataframe, grab that long-term mean value by its index that corresponds to the "doy" of the current row (done with long_term_means.index == df_bm.loc[row,"doy"]). Then we subtract that long-term mean from that corresponding datapoint.
```python
df_bm["diff"] = np.zeros(len(df_bm))
for row, index in df_bm.iterrows():
    ltm = long_term_means[long_term_means.index == df_bm.loc[row,"doy"]]
    diff = df_bm.loc[row,"data"] - ltm
    df_bm.loc[row, "diff"] = diff.values
```  

**Step 4**  
One thing is still missing: the threshold to define our datapoint as extreme! In this approach we define the thresholds for something to be extreme based on the "diff" column. We want to find those values, where the deviation from the long-term mean for that specific day of the year is larger than usual. Makes sense right? Again we can use the quantiles function to find the extremes of the differences:

```python
upper_thresh = df_bm["diff"].quantile(prob)
lower_thresh = df_bm["diff"].quantile(1-prob)
df_bm["extreme_high"] = df_bm["diff"] > upper_thresh, "data"
df_bm["extreme_low"] = df_bm["diff"] < lower_thresh, "data"
```  

Note: In the POT approach the quantiles where built from the whole dataset itself. Here, the quantiles are built from the array of deviations from the mean! Remember this in the exercise when you evaluate the results.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>

1. Go ahead and built a function for the block maxima method. You already got all the building blocks. Put them together and add the right function definition and return statement.  
2. After using POT and the BM, which method do you expect to yield more extreme values per year? How do you think the extremes of the two methods are different from each other?  
3. To compare the outcomes of the two functions you can plot the distributions of the extreme values together. In the "visualize_quantiles" method above you already have a function given that creates a distribution. Write a new function that builds distributions of the extreme values for the different methods and creates a plot. This can well be done by first creating an empty figure object and then looping through the different extremes-dataframes, calculating the distributions for each and adding a new trace. After the loop you can call the "fig.show()" to display the figure. A starter code is given below.

{::options parse_block_html="true" /}

<details><summary markdown="span">Starter Code ex. 3</summary>

```python
def visualize_extreme_distributions(dfs:list[pd.DataFrame], extr_high_col:str, extr_low_col:str, methods:list[str]):
    print("----")
    print("Printing extremes")
    colors = ["red", "blue", "green", "purple", "lightblue", "coral"]
    
    fig = go.Figure()
    for i,df in enumerate(dfs):
        method = methods[i]
        color = colors[i]
        #... calculate distributions and add new traces to  the figure
        # You can nicely visualize the different methods by giving them the 
        # same color but maybe differentiate low and high
        # extremes by using dashed and solid lines
        # Use the "method" variable to give the traces
        # labels (with the "name" parameter to tell them apart
        # in the legend)
```
</details>

<details><summary markdown="span">Solution Ex. 1</summary>

```python
# the full code for the block maxima method
def block_maxima(X:pd.Series, prob:float):
    df_bm = pd.DataFrame(index=X.index, data={
        "doy":X.index.day_of_year,
        "data":X.values,
        "data_14d_ma": X.rolling(window=14, min_periods=1, center=True).mean()
    })
    long_term_means = df_bm.groupby("doy")["data_14d_ma"].mean()

    df_bm["diff"] = np.zeros(len(X))
    for row, index in df_bm.iterrows():
        ltm = long_term_means[long_term_means.index == df_bm.loc[row,"doy"]]
        diff = df_bm.loc[row,"data"] - ltm
        df_bm.loc[row, "diff"] = diff.values
    
    upper_thresh = df_bm["diff"].quantile(prob)
    lower_thresh = df_bm["diff"].quantile(1-prob)
    df_bm["extreme_high"] = df_bm["diff"] > upper_thresh
    df_bm["extreme_low"] = df_bm["diff"] < lower_thresh
    return df_bm

```
</details>

<details><summary markdown="span">Solution Ex. 3</summary>

```python
def plot_extremes_distribution(dfs:list[pd.DataFrame], extr_high_col:str, extr_low_col:str, methods:list[str]):
    print("----")
    print("Plotting extreme distributions")
    colors = ["red", "blue", "green", "purple", "lightblue", "coral"]
    
    fig = go.Figure()
    for i,df in enumerate(dfs):
        method = methods[i]
        color = colors[i]
        
        extr_highs = df.loc[df[extr_high_col] == True, "data"]
        extr_lows = df.loc[df[extr_low_col] == True, "data"]
        y_high = stats.norm.pdf(extr_highs.sort_values(), extr_highs.mean(), extr_highs.std()) # This function creates the y-values of the normal distribution given our data, the mean and the standard deviation
        y_low = stats.norm.pdf(extr_lows.sort_values(), extr_lows.mean(), extr_lows.std()) # This function creates the y-values of the normal distribution given our data, the mean and the standard deviation
        fig.add_traces(
            go.Scatter(
                x=extr_highs.sort_values(),
                y=y_high,
                name = f"{method} extreme highs",
                mode = "lines",
                line_color = color
                )
        )
        fig.add_traces(
            go.Scatter(
                x=extr_lows.sort_values(),
                y=y_low,
                name=f" {method} extreme lows",
                mode = "lines",
                line_color = color,
                line_dash = "dash"
                )
        )
    fig.update_layout(template="simple_white") # <- not neccessary, I just like it!>
    fig.show()
```
{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>
---

### 3.3. Moving Average Method (MA)

The final method we will look at is the moving average method. As the name already states, here the extremes are detected on a more temporally constrained basis, the moving average around each datapoint. Take a look at the code block for the block-maxima method. Everything we need for the moving average method is already in there. This time, try to write the method all by yourself. It is really not hard. You just need to figure out, which data you need to subtract to get the "diff" column right. As a little hint: You don't need the day_of_year information here anymore at all.  

----

### Exercises

1. Think about how using a smaller time reference window might affect the extreme value detection. Would you expect extreme values in this approach to be more or less frequent than in the block averaging method? Then run the detection function and save the output in a new variable.
2. You have now run all three methods. How do you think does the distribution look for the moving average? Use your plotting function from before to check your hypothesis. 
3. Change the parameter rollmean_period of the extreme detection function with the MA method to 365 and pass that output to the ```Plot_All_Extremes()``` function. How do you explain the output in comparison to the other methods?  

----
