---
title: "3. INTERPOLATION AND GAP FILLING"
layout: default
nav_order: 5
---

## **Interpolation and Gap Filling**

In this exercise we look at interpolation and the rather time-series specific topic of data-gap filling.  
In time-series data we often have gaps due to a variety of reasons. They can result from instrumental issues or maintenance times or unfavorable weather conditions which leads to data being discarded. These data gaps can be filled with statistical methods.

In this lesson exercises are not completely separated from the content. Just follow along, grab the code and in some parts you will get snippets to run and fiddle with yourself.

Before we start plotting data we will see, how we can deal with missing values which are already handled by the institution
measuring the data, e.g. the DWD. For example it is common that the data is included with a specific placeholder value, which
we first need to handle.

### 1. Loading and  converting data:
We will use some data I have prepared in a way that you might find it in an online data portal.  
[Download the file here](https://nicbehr.github.io/new_bai_data_analysis/assets/data/dwd_diepholz_1996_2023_missing_placeholders.parquet)

To test some things we will work with the air temperature column "tair_2m_mean" here.
There are several issues when we have a missing-data-placeholder like that. Try two things:

 <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Look at a quick express plot of the data. Is that a meaningful representation?
Then try to resample this data to daily values. Plot the data, do the values make sense?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
df_dwd = pd.read_parquet(<path to file>)
df_dwd["date_time"] = pd.to_datetime(df_dwd["date_time"])

fig = px.scatter(df_dwd, y = "ta_2m_mean")
fig.show()
```
Plotting the data with the missing value placeholder makes the data barely readable:  
![Image of data with missing values](/assets/images/python/3/missing_values.png)

```python 
df_dwd_daily = df_dwd.resample(rule="d", on="date_time").mean()

fig = px.scatter(df_dwd_daily, y = "tair_2m_mean")
fig.show()
```
When averaging the values, the -999.99 values are taken into account leading
to unrealistic results:  
![Image of poorly average data](/assets/images/python/3/bad_averaged_data.png)

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

Lets look at the first way to solve this issue. We have to find the rows, where
the values are the placeholder value. You can identify these rows
by grabbing the right-row indices of the dataframe with a condition:  
```python
# we find the indices in the column of air temperature, where
# the values are -999.99:

#                                     |  here we find the rows we need:    | column we need|
#                                     | those where the row tair is -999.99|
#                                                        |                      |
#                                                        v                      v
indices_of_missing_values = df_dwd.loc[df_dwd["tair_2m_mean"] == -999.99, "tair_2m_mean"].index
```

We can then go ahead and replace these
values with a special type, a "NaN"-value. NaN stands for Not a Number and
represents specifically missing numeric values. This object is provided by the 
numpy package and works nicely with pandas:

```python
import numpy as np # do this if you have not already imported numpy
df_dwd.loc[indices_of_missing_values, "tair_2m_mean"] = np.NaN
```

Now the irritating -888.88 values are replaced and you can easily plot and resample
the data in a meaningful manner:

```python
fig = px.scatter(df_dwd, y="tair_2m_mean")
fig.show()

df_dwd_daily = df_dwd.resample(rule="d", on="date_time").mean()
fig_daily = px.scatter(df_dwd_daily, y="tair_2m_mean")
fig_daily.show()

```

### 2. Gap Filling, interpolation and modelling

In the next part we will discuss how we can work with timeseries that have gaps of different sizes.
This is a regular task when working with long-time observations and there are a couple of options,
depending on what data is available to you and what is the final evaluation goal you have in mind.

#### 2.1: Simple linear interpolation

You do basic interpolation in your every day live. You want to bake a cake and only find a receipe for an 8 person cake,
but only 3 friends are coming over for cake time. In the receipe you have to use 1 kg of flour. Intuitively you can 
see that since you will only be four at the table, you can alter the receipe and only use 500 g of flour.
And already did you do some interpolation! 
What you easily did right away in your head could be mathematically formulated as:
y = 125 * x
where y is the amount of flour in grams and x is the number of people eating cake.

The formula for an interpolation between two points (x1,y1) and (x2,y2) at a specific point
(xn, yn) is:  

$$
yn = y1 + \frac{(y_{2}-y_{1})}{(x_{2}-x_{1})} * (x_{n} - x_{1})
$$

We simply construct a straight line where y1 is our y-intercept, the slope is derived 
from the two points with the well known slope-formula 

$$
m = (y2-y1)/(x2-x1)
$$ 

and our x value on this constructed line is difference between the point we want to look at minus the starting point

Note that in this form of y = mx + b we only have one x which we use to explain our y-value. We have one "predictor".
Using only one predictor gives us a so called simple linear regression. This is a super simple form of interpolation 
and of course leaves a lot of information aside. 

Lets look at a simple example of how to actually do linear interpolation in Python:  
  
First we create a data set to play with. We create a simple running index from 1 to 11 and some 
made up y-values. We make one array in which all values are present and a second in which some
values are missing. :

```python
index = [1,2,3,4,5,6,7,8,9,10,11]
data = {
    "full_data" : [1,2,0,13,4,10,19,15,13,21,27],
    "missing_data" : [1,2,0,np.NaN,4,np.NaN,19,15,13,np.NaN,27]
}
data = pd.DataFrame(index = index, data = data)
```

lets take a quick look at the two datasets:

First we create as simple plot to look at the characteristics of the data. Lets make a 
quick little function to keep a bit of styling:  

```python
import plotly.express as px
def scatter_plot_interp(data, columns:list[str], show=True):
    fig = px.scatter(data, y=columns)
    fig.update_traces(marker_size=10)
    fig.update_layout(template="simple_white")
    if show:
      fig.show()
    return fig
scatter_plot_interp(data, ["full_data", "missing_data"])
```

Since the two plots overlay each other, you can see the "missing" values in blue and all the ones
in the reduced dataset in red.  

To do a linear interpolation between each adjacent points you can use a numpy function, np.interp() or a built-in pandas method.
You can find its documentation [here](https://numpy.org/doc/stable/reference/generated/numpy.interp.html).  
The function takes 3 main arguments: 
1. The x-coordinates for which the data shall be interpolated
2. The x-coordinates of the input data
3. The y-values of the input data

The catch with the numpy function however is that the function will return NaN if there are NaN-values in the input arrays. 
The pandas function is a lot easier, but we will have to deal more with this problem of getting rid of NaN values later when we use other models,
so we can practice getting rid of NaN data in our training data now anyways.  
Specifically we will do the following:  
- find indices of NaN values
- find indices of non-NaN-values
- first argument is where to interpolate, so provide the indices of the NaN values
- second and third arguments are the x and y values of the adjacent non-NaN values,
so provide the index and the y-values at the non-NaN indices
 
```python
# 1. get indices of missing and present points:

indices_of_missing_points = data.loc[data["missing_data"].isna()].index
indices_of_present_points = data.loc[data["missing_data"].notna()].index

# 2. interpolate missing values and store them in 
# a new column in the dataframe. We can either do this with
# the numpy function np.interp:
data.loc[indices_of_missing_points,"interpolated_data"]= np.interp(indices_of_missing_points, 
          data.loc[indices_of_present_points,"missing_data"].index, 
          data.loc[indices_of_present_points,"missing_data"])

# the pandas approach is much easier to use and is simply:
data["interpolated_data"] = data["missing_data"].interpolate()
```

In this approach all we did was to draw straight lines between adjacent points. As you see, for the first point
the prediction was rather poor, the other two where pretty well reconstructed:

```python
fig = scatter_plot_interp(data, ["full_data", "interpolated_data"])
```

However, with this approach we leave all the information the other points give us about the data aside. 
Imagine for example that you have a timeseries where you measure temperature at midnight and at 12AM.
If one datapoint was missing, you would connect the two night time temperatures and interpolate the daytime
temperature way off.

A simple measure of how well our model performed is to look at the root mean squared error.

$$ 
RMSE = \sqrt{\overline{(y[i] - ypred[i])^2}} 
$$

where y is the true value and ypred is the predicted y value. You simply compute the model error for 
each datapoint and square them to avoid counter balancing of negative and positive errors. Then you take 
the mean of these values and finally take the square root, to get back into your data range.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Use your knowledge of pandas and numpy to write a function that returns the RMSE</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
def get_RMSE(y_true, y_predicted):
    RMSE = np.sqrt(np.mean((y_true - y_predicted)**2))
    return RMSE

y_true = data.loc[indices_of_missing_points, "full_data"]
y_predicted = data.loc[indices_of_missing_points, "interpolated_data"]
RMSE = get_RMSE(y_true, y_predicted)
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>
#### 2.2: Simple linear models

Another approach is be to create a linear model that builds not only on the two points adjacent to the one we want to know,
but rather the whole of the dataset that we have available.

So what we want to achieve, is to find a function that constructs our unknown data points based on the data we have
available in the best possible way. That means, that we want to have as little errors in our model as possible. 
The error is usually measured as the "sum of squared errors" (SSE) which is the total of distances between true values 
and the predicted values. We square it to avoid negative and positive values counterbalancing each other. 

Looking at an array of n data points we can write  

$$ 
SSE = \sum_{i=1}^n (y(i) - b - m * x(i))^2 
$$ 
  
y(i) is the true y value at the predicted point, b is the y-intercept of the linear model, 
m is the first coefficient of the linear model and x(i) is the x-value at the predicted point. 

Since we want to find the straight line, that MINIMIZES the SSE, we call a procedure like this
a "minimization problem" and specifically the estimation of this line is called a "least squares estimation".

In the easiest way of fitting a linear model to such a dataset, it all depends on the mean of our dataset.
To derive the model parameters we can use the following relations where we replace b with alpha and m with beta
(as that is the general standard). Also we will now denote the predicted y-value with a ^ on top of that, which is
the common standard in literature. Sometimes this is also referred to as y_hat.  

$$ \hat{y}_{i} = \alpha + \beta * x_{i} $$

$$ \alpha = \bar{y} - (m \bar{x}) $$

$$ \beta = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} $$

If we would do it by hand, we would simply plug in all the numbers we have into the expression for beta and
use the result to derive our alpha

But we are working with Python so we will now introduce an awesome modelling and machine-learning library called
"scikit-learn".  
Scikit-learn has a huge amount of model-packages available, from simple linear regression all the way advanced statistical
regressions, classifications and analysis tools. [The documentation is also quite nice and extensive!](https://scikit-learn.org/stable/index.html)  
We will make use of scikit-learn to fit a simple linear regression model to our data. However to do so we need to some tweaking of our data.
Especially two things are important:  
1. Scikit learn can not work with NaN-data. That means we need to filter these out of the data we feed to the linear model
2. The model needs two-dimensional data. A list like [1,2,3] is one-dimensional and does not work with creating linear models.
Instead we have to bring it to a form of [[1],[2],[3]] etc. We can do this using the "reshape" function.

```python
from sklearn.linear_model import LinearRegression
# Scikit learn is quite object oriented. That means,
# we imported a Class called "LinearRegression", which contains
# all the following functions to work with the model.

# Step 1: instantiate the class
linearModel = LinearRegression()

# Step 2: Fit the model. This is the process of 
# feeding our known data to the model and tweaking the
# parameters so that the prediction error gets minimized.
# However, sklearn can not work with NaN values, so again
# we need to leave them out in the fitting:

# --- repetition from before:
indices_of_present_points = data.loc[data["missing_data"].notna()].index
x = data.loc[indices_of_present_points,"missing_data"].index.values.reshape(-1,1)
y = data.loc[indices_of_present_points,"missing_data"].values.reshape(-1,1)
# --- fitting the model:
linearModel.fit(x,y)

# Step 3: Check how well our model performed! sklearn has an inbuilt
# function for it called "score". It returns the R^2 value for 
# the true values and the values predicted by the model:
linearModel.score(x,y)
```  
We can obtain the paramters of the linear model, that scikit-learn has created for us:

```python
m = linearModel.coef_
b = linearModel.intercept_
print(f"Linear equation: {m}*x+{b}")
```

The last thing left to do is to use this model to predict our missing values.
All we need to do is use the models "predict()" function and give it the 
indices we want to prediction for:

```python
# first we create a new column consisting of NaN values
data["sklearn_prediction"] = np.NaN
# then we replace the values in the missing rows with our model prediction:
linear_prediction = linearModel.predict(indices_of_missing_points.values.reshape(-1,1))
data.loc[indices_of_missing_points, "sklearn_prediction"] = linear_prediction
```

We can look at out interpolated values by plotting them as red dots together with our reduced dataset:
```python
fig_linmod = scatter_plot_interp(data, ["full_data", "sklearn_prediction"], show=False)
```
As you can see, the linear model already performs a bit better than the simple linear interpolation does. 
We can visualize the linear regression line by predicting the full array of x-values and plotting
the result as a line:

```python
data["yhat_full"] = linearModel.predict(data.index.values.reshape(-1,1))
fig_linmod.add_traces(
    go.Scatter(
        x = data.index,
        y = data["yhat_full"],
        mode="lines",
        name="linear regression line"
    )
)
fig_linmod.show()
```
Finally we need to look at statistical metrics to find out, how well our linear model performed.
Luckily we can easily get a whole range of such metrics from the sklearn.metrics package. 
Lets define a simple function to grab a bunch of metrics at once:

```python

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('r^2: ', round(r2,4))
    print('MAE: ', round(median_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
```

A few things we can take from this are:  
**a)** r^2 is 0.8351, which is the ratio of the sum of squared errors divided by the sum of squared
deviations from the mean. You can say that r^2 is a measure of how much of the variance in the original
data is reflected by the model. In this case, as our model is just a line, the amount of variance captured in the model
stems from the linear trend that is inherent in the original data.

Whether an r^2 is reflective of a good correlation depends heavily on the application. If you are a social scientist and work on 
voter behaviour an r_square of 0.65 may be spectacularly good. If you want to calibrate your measurement device
and the reference and measured values have an r^2 of less than 0.85 you might want to check it again...

**b)** The mean squared error (MSE) is exactly that: we calculate the distance from each datapoint to its predicted counterpart,
and to avoid negative errors counterbalancing positive ones, we square them. Then we take the mean of all errors. Due to the squaring
the errors get quite high and are not directly interpretable. That is why we take the square root of the squared errors and get to the
"root mean square error" (RMSE). This is an error very often reported in model performance evaluation, also often used in scientific papers.

**c)** Lastly the median absolute error (MAE) is a different performance metric that gets shown not as often, but is still very useful.
For the MAE, we also calculate each error, take the absolute of it (make negative values positive) and then grab the median value,
so the one that sits right in the middle of all datapoints. Because we take the median instead of the mean, this metric is insensitive
to outliers. If we have very low errors, but then a few extremely high ones (or the other way around), the mean value can be skewed
while the median would not change.  

We wont go much deeper into statistical metrics here. But as you can see, this model does represent certain characteristics
of the data regarding its variance (judging by the rsquare of > 0.8)
but has a pretty high average error of more than 4 while we are in a 
domain of data that only reaches from 1 to 27.

### Part 2.3: Multiple linear models

Lets look at another way to make our models a bit more flexible
So far we created a linear model with only one parameter. Obviously that did not catch all of the variance in our 
data. In reality we often have more data at hand which can help us explain the measure of interest.
For example to fill gaps in temperature data instead of only using the indices to predict, we could add variables
such as the incoming radiation or the relative humidity of the air.

Lets continue working with the dwd data we used before. Load it just like we did in the previous exercises

When we try to simply interpolate with the pointwise linear interpolation, 
you will see that we get a pretty uninformed output.

We will now create a more sophisticated model to reconstruct our missing data. However, this time we have a whole dataset
of predictors to choose from.  
Since we want to fill a gap in temperature data, we need to find predictors that are well correlated with 
temperature. To figure out which ones are suitable we can make use of the correlation matrix.  
A correlation matrix is a normalized form of a covariance matrix. The values vary between -1 and 1. 
A value of 1 signals a perfect positive, -1 a perfect negative correlation. 0 means that the two variables are not
correlated at all. With pandas you can get the full correlation matrix with all variables with the .corr() function:

```python
# lets look at the correlation matrix
df_dwd.corr()
# you can plot and explore it with plotly.
# The interactivity is really handy here:
px.imshow(df_dwd.corr()).show()
# To get all correlations with tair_2m_mean we have to index it:
df_dwd.corr()["tair_2m_mean"]
```
Try to figure out, which variables could be suitable to fill the gaps in our data from the below table.

Now we can go ahead and start building our multivariate model! Let go!
Before we really start plugging the data into the model we need to do a bit of 
preparation:  
Since the model has to be fit with data where all the predictors we want are present AND
we have observation data of our target variable to train the model on, we first need to find that
data. We can do that easily by dropping the rows, where these columns are na with "dropna()":

```python
df_dwd_noNA = df_dwd.loc[:,["date_time","SWIN","rH","tair_2m_mean"]].dropna()
```

Now we want to split these into the data we use as predictors (y) and the data we want to 
predict (x, also called the "predictand"):

```python
x = df_dwd_noNA.loc[:,["SWIN","rH"]]
y = df_dwd_noNA.loc[:,["tair_2m_mean"]]
```

Finally one last very important step is that we need to split our available data into two parts:
a training and a testing dataset. The training data will ONLY be used for creating (or "fitting")
the model. To test the performance of the model, we keep a fraction of the available data out 
of the training set. That way we can predict the testing data and compare it to the real results.
We are working with some artificiallly created gaps in the data here, but in real life you would 
otherwise have no way to test, how well your model actually predicts data.  
Additionally to splitting the data, the training datasets also get shuffled. That makes the 
model more robust in extrapolating it to unknown data.  

It is extremely important to do this split, because you can never test a model on data that it 
has already seen during its training phase. That would skew your results and make it look better than
it actually is.  
Luckily, because this is such a common task to do, scikit learn has us covered with a very simple function
to do the splitting:
```python
from sklearn.model_selection import train_test_split 
# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	x, y, test_size=0.3, random_state=42) 
```

Now that we have our final training and testing datasets ready for use, we can go ahead and fit our model!

```python
linearModel = LinearRegression()
# Only training data used for fitting:
linearModel.fit(X_train,y_train)
# Only testing data used for the score:
linearModel.score(X_test,y_test)

# You can plot the prediction for the testing period
# as a scatter plot to get an idea of the spread
# of the errors. Put true values on one axis and predicted on the other:
y_hat_ml = linearModel.predict(X_test).reshape(1,-1)[0]
px.scatter(x=y_hat_ml,y=y_test["tair_2m_mean"]).show()
```
As you can see the score is roughly 0.36. That is not exactly great but does indicate
a weak correlation between predicted and true values. 

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
<h3> Exercise </h3>
<p >Do a linear interpolation and 1-D linear model prediction for this same data.
Do any of them perform equally good or better than the multiple regression? </p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
#------- interpolation:
# in order to interpolate value-by-value we need to
# first sort the previously randomized data:

y_train_sorted = y_train.sort_index()
y_test_sorted = y_test.sort_index()

interpolated_data = np.interp(
    y_test_sorted.index,
    y_train_sorted.index,
    y_train_sorted["tair_2m_mean"])

regression_results(y_test, interpolated_data)

#-------- 1-D linear model:
y = df_dwd_noNA.loc[:,["tair_2m_mean"]].values.reshape(-1,1)
x = df_dwd_noNA.index.values.reshape(-1,1)

from sklearn.model_selection import train_test_split 
# creating train and test sets 
X_train, X_test, y_train, y_test = train_test_split( 
	x, y, test_size=0.3, random_state=101) 

linearModel = LinearRegression()
linearModel.fit(X_train,y_train)
y_hat_linear = linearModel.predict(X_test)
regression_results(y_test, y_hat)
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

### 2.4: Machine Learning approaches (example Random Forests)
We already covered quite a lot of ground on how to deal with missing data by
- cleaning the raw data
- gap fill with interpolation, 1D-linear modeling and multiple linear regression

In this final part we will take a quick look at a more sophisticated type of model, the
Random Forests algorithm. Random Forest is a so-called decision-tree algorithm
and can be counted to the broad category of "machine-learning" methods.
The latter however is reeeeally a broad category, as it basically just describes that
the machine works through a minimization procedure on such a large amount of data, that
humans could not handle it manually, thus the machine is "learning" the optimization of
the model and can make predictions from it.  

Random Forests has proven to be quite effective in gap-filling applications in a 
variety of contexts and is available as part of the scikit-learn package. 
The purpose is for you to get an idea, how to implement such a sophisticated method 
and to hopefully get you excited about machine learning! We will not go
into the details of the actual method.

Lets dive right in and load the random forest regressor from scikit-learn. We use the regressor
because we work with time-series data. Random Forest also has a classification model, which is used
for categorical data (for example image-recognition, predicting an animal type from its traits etc...)

The great thing about scikit-learn is that most of the models work in the exact same way, no 
matter whether it is a simple linear model or a complex machine-learning approach.  
For example to run a model with simple default setting all we have to do is the following:  

```python
from sklearn.ensemble import RandomForestRegressor
  # I set n_estimators to 12 for a quick initial fit
  # we will go into the parameters a bit more later!
rf_model = RandomForestRegressor(random_state=42, n_estimators=12)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
y_hat_rf = rf_model.predict(X_test)
regression_results(y_test, y_hat_rf)
```
The predicting performance is still not that great. However, with a machine-learning approach we can feed 
some more data into the model and see, whether it improves the model.  
I will only give a very brief intro to random forests here, no need to memorize that. If you are interested,
you can also look the below youtube video for a very good short video on the method.  
https://www.youtube.com/watch?v=v6VJ2RO66Ag  
   
Very generally speaking you can say that this algorithm looks at your data and the predictors and it picks
a few of  the predictors, leaving others out.  
With this reduced set it trains a model. That means, it tries to find out under which circumstances in the 
predictors, the data has a certain value.   
In random forests, many of those models are trained and compared. Each with different predictors and trained on 
different amounts and points of training data.  
After building the model, you can use it to predict unknown values. Therefore, the predictor data for these 
unknown datapoints is fed into each of these models and the combined output from all of them is evaluated as the
final decision.

Lets try adding some more of our weather-data into the model and see whether it improves the performance:
```python
# Lets add the other weather-data columns into the predictor data as well.
# First we find the rows where all the predictors data and our observations
# are present:
df_dwd_noNA = df_dwd.loc[:,["date_time", "SWIN","rH", "pressure_air", "wind_speed", "precipitation", "tair_2m_mean"]].dropna()

# Now we split them into the x-values (predictors) and the y-values
# (predictand or target variable)
x = df_dwd_noNA.loc[:,["SWIN","rH", "pressure_air", "wind_speed", "precipitation"]]
y = df_dwd_noNA.loc[:,["tair_2m_mean"]]

# Now we go an split the data into training and testing data:
X_train, X_test, y_train, y_test = train_test_split( 
	x, y, test_size=0.3, random_state=101) 

# Finally we do the full pipeline of 
# creating the model, fitting it, and scoring:
rf_model = RandomForestRegressor(random_state=42, n_estimators=12)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
y_hat_rf = rf_model.predict(X_test)
regression_results(y_test, y_hat_rf)
# Aha, the model performs a bit better.
```

Finally, we will do some tweaking on the random forest paramters. Parameters are 
options given to the model, that define how it is set up. Here for example n_estimators
is one parameter we gave to the model so far.  
Maybe we can make the model perform even a bit better by increasing that value.
In order to do so, we better use hourly data, because as we make the model bigger,
the time it takes to fit the model gets substanitally larger.  
Lets first aggregate the data like before:  

```python
# mean for most data:
df_dwd_hourly_noNA = df_dwd_noNA.loc[:,["SWIN","rH", "pressure_air", "wind_speed","tair_2m_mean", "date_time"]].resample(rule="1h", on="date_time").mean().dropna()
# sum for precipitation data:
df_dwd_hourly_noNA["precipitation"] = df_dwd_noNA.loc[:,["precipitation", "date_time"]].resample(rule="1h", on="date_time").sum().dropna()
```

Now we can run a new model on the hourly data and e.g. set the n_estimators to 50.
Play around with the parameter a bit and see how the performance changes:
```python

x_hourly = df_dwd_hourly_noNA.loc[:,["SWIN","rH", "pressure_air", "wind_speed", "precipitation"]]
y_hourly = df_dwd_hourly_noNA.loc[:,["tair_2m_mean"]]

X_train, X_test, y_train, y_test = train_test_split( 
	x_hourly, y_hourly, test_size=0.3, random_state=101) 
rf_model = RandomForestRegressor(random_state=42, n_estimators=50)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
y_hat_rf = rf_model.predict(X_test)
regression_results(y_test, y_hat_rf)
# as you can see, the model performs yet another bit better.
```
<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Practice makes perfect! For the hourly data, see how the linear interpolation, 1D-linear model and 
multiple linear models perform compared to the random forest regression.</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
#------- preparation of  data:
# mean for most data aggregation:
df_dwd_hourly_noNA = df_dwd_noNA.loc[:,["SWIN","rH", "pressure_air", "wind_speed","tair_2m_mean", "date_time"]].resample(rule="1h", on="date_time").mean().dropna()
# sum for precipitation aggregation:
df_dwd_hourly_noNA["precipitation"] = df_dwd_noNA.loc[:,["precipitation", "date_time"]].resample(rule="1h", on="date_time").sum().dropna()

x_hourly = df_dwd_hourly_noNA.loc[:,["SWIN","rH", "pressure_air", "wind_speed", "precipitation"]]
y_hourly = df_dwd_hourly_noNA.loc[:,["tair_2m_mean"]]
X_train, X_test, y_train, y_test = train_test_split( 
	x_hourly, y_hourly, test_size=0.3, random_state=101) 

print("------- linear intrpolation:")
y_train_sorted = y_train.sort_index()
y_test_sorted = y_test.sort_index()
interpolated_data = np.interp(
    y_test_sorted.index,
    y_train_sorted.index,
    y_train_sorted["tair_2m_mean"])
regression_results(y_test_sorted, interpolated_data)

print("------- multiple linear regression:")
linearModel = LinearRegression()
linearModel.fit(X_train,y_train)
y_hat = linearModel.predict(X_test)
errors = (y_test - y_hat).iloc[:,0].values
regression_results(y_test, y_hat)

print("------- random forest:")
rf_model = RandomForestRegressor(random_state=42, n_estimators=50)
rf_model.fit(X_train, y_train.values.ravel())
rf_model.score(X_test, y_test)
y_hat_rf = rf_model.predict(X_test)
regression_results(y_test, y_hat_rf)
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>
For hourly data the linear interpolation still performs best.
However, the gaps we are interpolating thus far are rather small. As a last exercise, we will see how the methods
perform for longer gaps. Therefore I create a gap in the hourly dataset of a full day. We will then see how
the different methods perform in filling the gap:

```python
df_dwd_hourly_noNA = df_dwd_noNA.loc[:,["SWIN","rH", "pressure_air", "wind_speed","tair_2m_mean", "date_time"]].resample(rule="1h", on="date_time").mean().dropna()
df_dwd_hourly_noNA["precipitation"] = df_dwd_noNA.loc[:,["precipitation", "date_time"]].resample(rule="1h", on="date_time").sum().dropna()

# first lets create the 14-day long gap:
# I first extract the indices of some single day and safe them
indices_for_gap = df_dwd_hourly_noNA.iloc[505:529, :].index
# Now I make a copy of the original data to not mess it up
gapped_data_hourly = df_dwd_hourly_noNA.copy()
# Then I set all the values for tair in these indices to NaN
gapped_data_hourly.loc[indices_for_gap, "tair_2m_mean"] = np.NaN
# Finally I can extract the predictor and predictand columns with these indices:
x_hourly = gapped_data_hourly.loc[indices_for_gap,["SWIN","rH", "pressure_air", "wind_speed", "precipitation"]]
y_true = df_dwd_hourly_noNA.loc[indices_for_gap, "tair_2m_mean"]

#---- interpolation
interpolated_data = gapped_data_hourly["tair_2m_mean"].interpolate()
regression_results(y_true, interpolated_data[indices_for_gap])

#---- multiple linear regression:
y_hat_linear = linearModel.predict(x_hourly)
regression_results(y_true, y_hat_linear)

#---- Random Forest:
y_hat_rf = rf_model.predict(x_hourly)
regression_results(y_true, y_hat_rf)

```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Play around with the length of the gap and observe how the performance of the different methods changes.
Try to give an explanation and maybe formulate, when something like linear interpolation could be suitable and 
when it is better to rely on a more complex method.</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>
The Random Forest method works quite nicely on longer prediction windows. As long as there is 
a clear linear trend in the data, a simple interpolation might perform very well. However, if 
within a data gap a shift happens and for example a warm period comes around, the linear interpolation 
will quickly get worse in its prediction.
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>
As long as there is a clear linear trend in the data, a simple interpolation might perform very well. However, if 
within a data gap a shift happens and for example a warm period comes around, the linear interpolation 
can not capture that. 

Keep in mind that a model is by definition NEVER the actual truth. Its goal is to come as close as possible to the truth
while often times drastically reducing the complexity of the issue. In a real world scenario we would not
know that our final modelled data is not the same as the true data. Therefore all we can do is create and test our models
to the best of our knowledge and be honest about what they are capable of doing and what their shortcomings are!
