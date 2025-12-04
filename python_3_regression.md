---
title: "3. REGRESSION"
layout: default
nav_order: 4
---
# **Regression**

## Table of Contents

- [1. What's Regression All About?](#chapter-1-whats-regression-all-about)
- [2. Simple Linear Regression](#chapter-2-simple-linear-regression)
- [3. Multiple Regression](#chapter-3-multiple-regression)
- [4. Machine Learning with Random Forests](#chapter-4-machine-learning-with-random-forests)
- [5. Filling Gaps in Time Series](#chapter-5-filling-gaps-in-time-series)

Welcome! This tutorial will walk you through regression analysis - one of the most useful tools you'll encounter for making sense of ecological data. We'll start from the basics and work our way up to more advanced machine learning methods.
Don't worry if statistics isn't your strong suit. We'll take it step by step, and by the end you should feel comfortable applying these techniques to your own data.


---

# **1. What's Regression All About?**

## The Basic Idea

Here's the thing: as ecologists, we're constantly trying to figure out what drives the patterns we observe. Why are there more species in some places than others? What makes trees grow faster? How does temperature affect animal behavior?

Regression gives us a way to quantify these relationships. Instead of just saying "warmer temperatures seem to increase growth," we can say "for every 1°C increase in temperature, tree ring width increases by 0.15 mm." That's powerful stuff.

At its core, regression asks: **how does one thing change when another thing changes?**

## A Quick Example

Let's say you're studying tree growth across a temperature gradient. You measure tree ring widths at different sites:

| Mean Annual Temperature (°C) | Tree Ring Width (mm) |
|------------------------------|----------------------|
| 8                            | 1.2                  |
| 10                           | 1.8                  |
| 12                           | 2.4                  |
| 14                           | 2.9                  |
| 16                           | 3.2                  |

You can see there's a pattern - warmer sites have wider rings. But how strong is this relationship? Can we predict growth at a site with 11°C mean temperature? Regression helps us answer these questions.

## What Can Regression Achieve?

In ecological research, regression is useful for:

**Making predictions** - You've measured carbon flux at 20 sites, but you want to estimate it across the whole landscape. Regression lets you predict values at unmeasured locations based on environmental variables you can get from satellite data.

**Understanding relationships** - Does nitrogen addition actually increase plant biomass? By how much? Is the effect statistically significant or could it just be noise?

**Figuring out what matters** - When you have 15 environmental variables that might explain species richness, regression helps you sort out which ones are actually important.

**Supporting management decisions** - If you know how much habitat area affects population size, you can make informed recommendations about reserve design.

## Terminology

Before we dive in, let's get our vocabulary straight. Different fields use different terms for the same things, which can be confusing.

**Target variable**
- You can also call it response variable, dependent variable, outcome
- The variable you're trying to predict or explain
- We usually call it **y**
- Examples: species richness, biomass, survival rate, carbon flux

**independent variables**
- Can also be termed predictors, features, independent variables, explanatory variables
- The variable you use to make predictions or explain target variables 
- We call these **x** (or x₁, x₂, etc. when there are several)
- Examples: temperature, precipitation, soil pH, elevation

**The model:**
- This is the mathematical equation that describes how x relates to y
- General form: y = f(x) + error
- The "error" part is important - it acknowledges that our model won't be perfect

**Coefficients:**
- These are the numbers in our model that define the relationship
- In a simple model like y = 3 + 2x, the "3" is the intercept and "2" is the slope
- We estimate these from our data

**Residuals:**
- The difference between what we observed and what our model predicted
- Small residuals = good model fit
- Patterns in residuals = something's wrong with our model

## How Does Regression Actually Work?

The basic process goes like this:

1. **Choose a model type.** Are you assuming a straight line relationship? A curve? Multiple predictors?

2. **Fit the model to your data.** This means finding the coefficient values that make your predictions as close to the observations as possible.

3. **Check if it worked.** Look at how well the model fits, whether the assumptions are met, and whether the results make ecological sense.

The most common approach for step 2 is called "least squares" - we find the coefficients that minimize the sum of squared differences between observed and predicted values. We square the differences so that positive and negative errors don't cancel out.

## Evaluating Your Model

How do you know if your model is any good? A few key metrics:

**R² (R-squared)**: This tells you what fraction of the variation in your data is explained by the model. An R² of 0.7 means your model explains 70% of the variance. What's "good" depends entirely on your system - in controlled experiments 0.9 might be expected, while in field ecology 0.3 might be excellent.
<div>$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} $$</div>
Where:

$y_i$ is the observed value
$\hat{y}_i$ is the predicted value  
$\bar{y}$ is the mean of observed values
$SS_{res}$ is the sum of squared residuals
$SS_{tot}$ is the total sum of squares


RMSE (Root Mean Square Error): This is the average size of your prediction errors, in the same units as your response variable. An RMSE of 2.5°C for a temperature model means your predictions are typically off by about 2.5 degrees.
<div>$$ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} $$</div>
MAE (Mean Absolute Error): Similar to RMSE but less sensitive to outliers. Useful when you have some weird extreme values in your data.
<div>$$ MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| $$</div>
The difference between RMSE and MAE? RMSE penalizes large errors more heavily because of the squaring. If you have a few really bad predictions, RMSE will be much higher than MAE. This can be useful for detecting outliers or problematic predictions.

---

# **2. Simple Linear Regression**

Alright, let's actually do some regression. We'll start with the simplest case: one predictor, one response, straight line relationship.

## The Model

Simple linear regression fits this equation:

**y = β₀ + β₁x**

Where:
- β₀ is the intercept (value of y when x is zero)
- β₁ is the slope (how much y changes for each unit increase in x)

That's it. We're just fitting a line through our data points.

## Let's Try It: Tree Growth and Temperature

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Here's some tree ring data from a temperature gradient
np.random.seed(42)

temperature = np.array([6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
ring_width = 0.8 + 0.15 * temperature + np.random.normal(0, 0.3, len(temperature))

tree_data = pd.DataFrame({
    'temperature_C': temperature,
    'ring_width_mm': ring_width
})

print("Our data:")
print(tree_data)
```

Now let's fit a regression. I'll show you both the manual calculation and the easy way with scikit-learn.

### The Math (for those who want it)

```python
# Calculate means
temp_mean = np.mean(temperature)
growth_mean = np.mean(ring_width)

# Slope formula
numerator = np.sum((temperature - temp_mean) * (ring_width - growth_mean))
denominator = np.sum((temperature - temp_mean) ** 2)
slope = numerator / denominator

# Intercept
intercept = growth_mean - slope * temp_mean

print(f"Slope: {slope:.4f} mm per °C")
print(f"Intercept: {intercept:.4f} mm")
```

### The Easy Way

In practice, you'll almost always use a library:

```python
from sklearn.linear_model import LinearRegression

# sklearn wants 2D arrays, hence the reshape
X = temperature.reshape(-1, 1)
y = ring_width

model = LinearRegression()
model.fit(X, y)

print(f"Slope: {model.coef_[0]:.4f} mm per °C")
print(f"Intercept: {model.intercept_:.4f} mm")
print(f"R-squared: {model.score(X, y):.4f}")
```

### What Do These Numbers Mean?

The slope of about 0.15 tells us that for every 1°C increase in mean annual temperature, tree ring width increases by roughly 0.15 mm. That's the key ecological finding here.

The intercept (around 0.8) would be the predicted ring width at 0°C. This doesn't make much biological sense - trees don't really grow at 0°C - but mathematically we need it to define our line.

The R² of around 0.75 tells us that temperature explains about 75% of the variation in ring width. Not bad! The remaining 25% is due to other factors we haven't measured (soil quality, genetics, competition, etc.) plus measurement error.

### Visualizing the Fit

Always plot your regression. Numbers alone can be misleading.

```python
# Get predictions
predicted = model.predict(X)

fig = go.Figure()

# Data points
fig.add_trace(go.Scatter(
    x=temperature, y=ring_width,
    mode='markers', name='Observations',
    marker=dict(size=10, color='forestgreen')
))

# Regression line
fig.add_trace(go.Scatter(
    x=temperature, y=predicted,
    mode='lines', name='Fitted line',
    line=dict(color='darkgreen', width=2)
))

fig.update_layout(
    title='Tree Growth vs Temperature',
    xaxis_title='Mean Annual Temperature (°C)',
    yaxis_title='Ring Width (mm)',
    template='simple_white'
)
fig.show()
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Try It Yourself </h3>
<p>Create a dataset for grassland productivity (kg/ha) vs annual precipitation (mm). 
Fit a regression and interpret the slope. What does it tell you about water limitation?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Precipitation gradient
precip = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])
# Biomass increases with rainfall (with some noise)
biomass = 500 + 3.5 * precip + np.random.normal(0, 150, len(precip))

model = LinearRegression()
model.fit(precip.reshape(-1, 1), biomass)

print(f"Slope: {model.coef_[0]:.2f} kg/ha per mm rainfall")
print(f"R-squared: {model.score(precip.reshape(-1,1), biomass):.3f}")

# The slope tells us that each additional mm of rainfall 
# gives us about 3.5 kg/ha more biomass. This is essentially
# a measure of rainfall use efficiency for this grassland.
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

## A Word of Caution

Simple regression is great, but it has obvious limitations. Ecological systems are complex - tree growth isn't just about temperature. There's precipitation, soil nutrients, competition, pests, genetics... 

Also, the relationship might not be linear. Many ecological relationships have thresholds or optima. Trees don't just keep growing faster forever as temperature increases - at some point it gets too hot.

That's where multiple regression and machine learning come in.

---

# **3. Multiple Regression**

In the real world, ecological responses depend on many factors simultaneously. Multiple regression lets us include all of them in one model.

## Why Go Multiple?

Think about what controls ecosystem carbon exchange:
- Light drives photosynthesis
- Temperature affects both photosynthesis and respiration
- Soil moisture determines water availability
- Humidity influences stomatal conductance

If we only model carbon flux against temperature, we're missing most of the picture. Multiple regression lets us ask: "what's the effect of temperature, *after accounting for* light, moisture, and humidity?"

## The Model

We extend our simple model to include multiple predictors:

**y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ...**

Each coefficient now tells us the effect of that variable *while holding the others constant*. This is crucial for interpretation.

## Example: Forest Carbon Flux

Let's build a model for net ecosystem exchange (NEE) - the balance between carbon uptake and release.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

np.random.seed(42)
n = 300

# Environmental drivers
solar_rad = np.random.uniform(100, 900, n)       # W/m²
air_temp = np.random.uniform(5, 30, n)           # °C
soil_moisture = np.random.uniform(10, 50, n)     # %
vpd = np.random.uniform(0.5, 3, n)               # kPa (vapor pressure deficit)

# Carbon flux - negative means uptake, positive means release
# These relationships are roughly realistic
nee = (
    5 -                           # baseline respiration
    0.015 * solar_rad +           # light drives uptake
    0.3 * air_temp +              # warmth increases respiration  
    -0.1 * soil_moisture +        # moisture supports uptake
    2.0 * vpd +                   # high VPD closes stomata, reduces uptake
    np.random.normal(0, 1.5, n)   # noise
)

flux_data = pd.DataFrame({
    'solar_radiation': solar_rad,
    'air_temperature': air_temp,
    'soil_moisture': soil_moisture,
    'vpd': vpd,
    'nee': nee
})

print("Dataset preview:")
print(flux_data.head())
```

### Check Correlations First

Before modeling, it's always good to see how variables relate to each other:

```python
print("\nCorrelations with NEE:")
print(flux_data.corr()['nee'].sort_values())
```

This gives you a first sense of which variables might be important predictors.

### Fitting the Model

```python
# Prepare our data
X = flux_data[['solar_radiation', 'air_temperature', 'soil_moisture', 'vpd']]
y = flux_data['nee']

# Always split into training and test sets!
# This is how we honestly evaluate our model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Look at the coefficients
print("Model coefficients:")
print(f"  Intercept: {model.intercept_:.3f}")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")
```

### Interpreting the Results

This is the important part. What do these numbers actually mean?

```python
print("""
What the coefficients tell us:

Solar radiation (-0.015): Each additional W/m² of radiation 
  decreases NEE by 0.015 µmol/m²/s. Negative because more light 
  means more photosynthesis, which is carbon uptake.

Air temperature (+0.30): Each degree warmer increases NEE by 
  0.30 µmol/m²/s. Positive because warmth stimulates respiration 
  more than photosynthesis in this model.

Soil moisture (-0.10): Wetter soils decrease NEE (more uptake).
  Makes sense - water stress limits productivity.

VPD (+2.0): High vapor pressure deficit increases NEE. When 
  the air is dry, plants close their stomata to conserve water,
  which also blocks CO2 uptake.
""")
```

### How Good Is Our Model?

```python
# Predict on test data (data the model hasn't seen)
y_pred = model.predict(X_test)

r2 = metrics.r2_score(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(f"R-squared: {r2:.3f}")
print(f"RMSE: {rmse:.2f} µmol/m²/s")

# Plot predicted vs observed
import plotly.express as px
fig = px.scatter(x=y_test, y=y_pred,
                 labels={'x': 'Observed NEE', 'y': 'Predicted NEE'})
# Add 1:1 line
fig.add_scatter(x=[y_test.min(), y_test.max()], 
                y=[y_test.min(), y_test.max()],
                mode='lines', name='1:1 line',
                line=dict(dash='dash', color='red'))
fig.update_layout(template='simple_white', 
                  title='How well does our model predict?')
fig.show()
```

### Does Adding Variables Help?

A natural question: does including more predictors actually improve our model?

```python
# Let's compare models with different numbers of predictors
results = []

# Just solar radiation
m1 = LinearRegression().fit(X_train[['solar_radiation']], y_train)
results.append({
    'Model': 'Solar only',
    'R²': m1.score(X_test[['solar_radiation']], y_test)
})

# Solar + temperature
m2 = LinearRegression().fit(X_train[['solar_radiation', 'air_temperature']], y_train)
results.append({
    'Model': 'Solar + Temp',
    'R²': m2.score(X_test[['solar_radiation', 'air_temperature']], y_test)
})

# All four
m4 = LinearRegression().fit(X_train, y_train)
results.append({
    'Model': 'All four',
    'R²': m4.score(X_test, y_test)
})

print(pd.DataFrame(results))
```

Usually, adding relevant predictors helps. But be careful - adding irrelevant variables can actually hurt your model's ability to generalize to new data (this is called overfitting).

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Try It Yourself </h3>
<p>Build a model predicting fish species richness in lakes based on: lake area, maximum depth, 
water temperature, and dissolved oxygen. Which factor matters most?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
np.random.seed(42)
n = 150

area = np.random.uniform(10, 5000, n)          # hectares
depth = np.random.uniform(2, 50, n)            # meters  
temp = np.random.uniform(10, 28, n)            # °C
oxygen = np.random.uniform(4, 12, n)           # mg/L

# Species richness - larger, deeper lakes with good oxygen have more species
richness = (
    5 + 0.005 * area + 0.3 * depth + 
    -0.1 * (temp - 18)**2 +  # optimum around 18°C
    1.5 * oxygen + 
    np.random.normal(0, 3, n)
).clip(min=1)

X = pd.DataFrame({'area': area, 'depth': depth, 'temp': temp, 'oxygen': oxygen})
y = richness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"R²: {model.score(X_test, y_test):.3f}\n")
print("Coefficients (larger absolute value = more important):")
for name, coef in sorted(zip(X.columns, model.coef_), 
                         key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name}: {coef:.4f}")

# Oxygen usually comes out strongest - makes ecological sense
# since it's essential for fish survival
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

## Limitations

Multiple regression is powerful but has some important limitations:

**It assumes linear relationships.** If temperature has an optimum (growth increases up to 25°C then decreases), a linear model won't capture that properly.

**It assumes additive effects.** The model says the effect of temperature is the same regardless of moisture levels. In reality, these factors often interact.

**It can struggle with many predictors.** If you have 50 environmental variables and only 100 observations, you're going to have problems.

These limitations bring us to machine learning approaches, which can handle more complexity.

---

# **4. Machine Learning with Random Forests**

Alright, now we're getting to the fun stuff. Machine learning sounds fancy, but the basic idea is simple: let the algorithm figure out the patterns in your data, rather than you specifying them in advance.

## Why Machine Learning?

Ecological relationships are often messy:
- Species have thermal optima, not just linear responses
- Effects of one variable depend on another (interactions)
- There might be thresholds or nonlinearities we didn't anticipate

Machine learning algorithms can discover these patterns automatically. You don't have to know the shape of the relationship beforehand.

## Decision Trees: The Building Block

Before we get to Random Forests, we need to understand decision trees. They're intuitive once you see how they work.

A decision tree is basically a flowchart of questions:
- Is temperature > 15°C?
  - Yes → Is rainfall > 500mm?
    - Yes → Predict high productivity
    - No → Predict medium productivity
  - No → Predict low productivity

The algorithm figures out which questions to ask and what thresholds to use by looking at your data.

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Using our carbon flux data from before
X = flux_data[['solar_radiation', 'air_temperature', 'soil_moisture', 'vpd']]
y = flux_data['nee']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a simple tree (max_depth limits how many questions it asks)
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Visualize it
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, fontsize=9)
plt.title('Decision Tree for Carbon Flux')
plt.tight_layout()
plt.show()

print(f"Tree R²: {tree.score(X_test, y_test):.3f}")
```

Decision trees are easy to interpret - you can literally see the rules. But they have a problem: they tend to overfit. A deep tree can memorize the training data perfectly but fail miserably on new data.

## Random Forests: Many Trees Are Better Than One

Random Forests solve the overfitting problem by building many trees and averaging their predictions. Each tree is a bit different because:

1. Each tree is trained on a random subset of the data (with replacement)
2. At each split, only a random subset of variables is considered

This randomness means individual trees might make mistakes, but averaging over hundreds of trees gives robust predictions.

```python
from sklearn.ensemble import RandomForestRegressor

# Fit a random forest
rf = RandomForestRegressor(
    n_estimators=100,    # number of trees
    max_depth=10,        # how deep each tree can go
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(f"Random Forest R²: {metrics.r2_score(y_test, y_pred):.3f}")
print(f"Random Forest RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f}")
```

## Comparing All Our Methods

Let's see how everything stacks up:

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

results = []

# Simple regression (just one predictor)
simple = LinearRegression()
simple.fit(X_train[['solar_radiation']], y_train)
pred = simple.predict(X_test[['solar_radiation']])
results.append({
    'Method': 'Simple regression',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

# Multiple regression
multi = LinearRegression()
multi.fit(X_train, y_train)
pred = multi.predict(X_test)
results.append({
    'Method': 'Multiple regression',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

# Decision tree
tree = DecisionTreeRegressor(max_depth=10, random_state=42)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
results.append({
    'Method': 'Decision tree',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

# Random forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
results.append({
    'Method': 'Random forest',
    'R²': metrics.r2_score(y_test, pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, pred))
})

print(pd.DataFrame(results).to_string(index=False))
```

## What's Driving the Patterns?

One of the nicest things about Random Forests is that they tell you which variables matter most:

```python
importance = pd.DataFrame({
    'Variable': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nVariable importance:")
print(importance.to_string(index=False))

# Plot it
fig = px.bar(importance, x='Importance', y='Variable', orientation='h',
             title='What drives carbon flux?')
fig.update_layout(template='simple_white', yaxis={'categoryorder': 'total ascending'})
fig.show()
```

This is ecologically valuable - it tells you which environmental factors to focus on in future research or monitoring.

## Tuning Your Forest

Random Forests have some settings (hyperparameters) you can adjust:

```python
# How many trees do we need?
results = []
for n_trees in [10, 50, 100, 200, 500]:
    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results.append({
        'Trees': n_trees,
        'R²': rf.score(X_test, y_test)
    })

print(pd.DataFrame(results))
# Usually performance plateaus around 100-200 trees
```

The main parameters to consider:
- **n_estimators**: More trees = better but slower. 100-500 is usually fine.
- **max_depth**: Deeper trees can capture more complexity but might overfit.
- **min_samples_leaf**: Requiring more samples per leaf prevents overfitting.

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Try It Yourself </h3>
<p>Build a Random Forest to predict plant species richness from soil pH, nitrogen, 
precipitation, and grazing intensity. Compare it to multiple regression. 
Which variables matter most for plant diversity?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
np.random.seed(42)
n = 200

soil_ph = np.random.uniform(4.5, 8.0, n)
nitrogen = np.random.uniform(10, 200, n)
precip = np.random.uniform(400, 1200, n)
grazing = np.random.uniform(0, 3, n)

# Non-linear relationships - this is where RF should shine
richness = (
    20 - 2*(soil_ph - 6.5)**2 +      # optimum pH around 6.5
    -0.0005*(nitrogen - 50)**2 +      # intermediate N best
    0.01*precip +                     # more rain helps
    5*grazing - 2*grazing**2 +        # moderate grazing best
    np.random.normal(0, 3, n)
).clip(min=1)

X = pd.DataFrame({
    'soil_ph': soil_ph, 'nitrogen': nitrogen,
    'precipitation': precip, 'grazing': grazing
})
y = richness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multiple regression
lr = LinearRegression().fit(X_train, y_train)
print(f"Multiple regression R²: {lr.score(X_test, y_test):.3f}")

# Random forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest R²: {rf.score(X_test, y_test):.3f}")

print("\nVariable importance:")
for name, imp in sorted(zip(X.columns, rf.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.3f}")

# RF should do better because of the non-linear relationships
# soil_ph and grazing are probably most important due to their 
# strong non-linear effects
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

## When to Use What?

Here's my rough guide:

| Situation | Method |
|-----------|--------|
| Simple exploratory analysis | Simple regression |
| Need interpretable coefficients | Multiple regression |
| Complex patterns, many variables | Random Forest |
| Small dataset (<50 samples) | Stick to regression |
| Need to explain to non-statisticians | Decision tree or regression |
| Just want the best predictions | Random Forest |

In practice, I often fit both multiple regression and Random Forest. The regression gives me interpretable coefficients, while Random Forest tells me if there's predictive signal I'm missing with the linear model.

---

# **5. Gap-filling in Time Series**

Now let's apply what we've learned to a practical problem: dealing with missing data in ecological time series.

## The Problem

If you've worked with field data, you know this frustration. Your sensor died for a week. The battery ran out during the coldest part of winter. Someone accidentally unplugged the datalogger.

Missing data is annoying because:
- You can't calculate annual totals or means
- It messes up time series analyses
- Some statistical methods can't handle NaN values

## Loading Messy Data

Real data often has placeholder values instead of proper missing data markers. Let's see an example:

```python
import pandas as pd
import numpy as np

# Load meteorological data
df = pd.read_parquet('./dwd_ahaus_1996_2023_missing_placeholders.parquet')
df["data_time"] = pd.to_datetime(df["data_time"])

print(f"Temperature range: {df['tair_2m_mean'].min():.1f} to {df['tair_2m_mean'].max():.1f}")
```

If you see -999.99 as the minimum, that's a placeholder for missing data - not an actual temperature! We need to fix this:

```python
# Replace placeholder with NaN
df.loc[df["tair_2m_mean"] == -999.99, "tair_2m_mean"] = np.NaN

# Now check
print(f"Missing values: {df['tair_2m_mean'].isna().sum()}")
```

## Method 1: Linear Interpolation

The simplest approach - just draw a straight line between known values:

```python
# Pandas makes this easy
df['temp_interp'] = df['tair_2m_mean'].interpolate(method='linear')
```

This works fine for short gaps. If temperature was 10°C at noon and 14°C at 2pm, it's reasonable to guess 12°C at 1pm.

But it fails badly for longer gaps. It can't capture the daily temperature cycle - if you have a 24-hour gap, linear interpolation will give you a flat line right through where the daily max and min should be.

## Method 2: Regression-Based Gap Filling

If we have other variables that were measured continuously, we can use them to estimate the missing temperatures:

```python
from sklearn.linear_model import LinearRegression

# Solar radiation and humidity are often available when temperature fails
# (different sensors)
predictors = ['SWIN', 'rH']

# Get data where everything is present (for training)
df_complete = df[['data_time', 'SWIN', 'rH', 'tair_2m_mean']].dropna()

X = df_complete[predictors]
y = df_complete['tair_2m_mean']

# Fit model
model = LinearRegression()
model.fit(X, y)

print(f"Gap-filling model R²: {model.score(X, y):.3f}")
```

Then we can predict temperature wherever we have radiation and humidity data:

```python
# Find rows where temp is missing but predictors exist
mask = df['tair_2m_mean'].isna() & df['SWIN'].notna() & df['rH'].notna()
df.loc[mask, 'temp_regression'] = model.predict(df.loc[mask, predictors])
```

## Method 3: Random Forest Gap Filling

For better accuracy, especially with complex patterns, Random Forest often wins:

```python
from sklearn.ensemble import RandomForestRegressor

# Use more predictors
all_predictors = ['SWIN', 'rH', 'pressure_air', 'wind_speed', 'precipitation']
df_complete = df[all_predictors + ['tair_2m_mean']].dropna()

X = df_complete[all_predictors]
y = df_complete['tair_2m_mean']

rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X, y)

print(f"Random Forest R²: {rf.score(X, y):.3f}")

# Check which predictors matter most for estimating temperature
for name, imp in sorted(zip(all_predictors, rf.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.3f}")
```

## Which Method When?

After working with a lot of gap-filled data, here's what I've found:

**Short gaps (a few hours):** Linear interpolation is usually fine. Temperature doesn't change that fast.

**Medium gaps (a day or two):** Regression with environmental predictors. This captures the daily cycle if you have radiation data.

**Long gaps (weeks+):** Random Forest or similar, but honestly... consider whether you should be filling such long gaps at all. Sometimes it's better to acknowledge the data is missing.

**General advice:**
- Always validate your gap-filling on data where you know the truth
- Flag gap-filled values in your final dataset
- Report the uncertainty or error in your gap-filled values
- Don't over-fill - sometimes missing data should stay missing

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Try It Yourself </h3>
<p>Compare linear interpolation vs. Random Forest for filling a 24-hour gap in temperature data.
Which method captures the daily cycle better?</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
# The key insight is that linear interpolation can't capture 
# diurnal patterns, while Random Forest (using radiation as 
# a predictor) can.

# For a 24-hour gap:
# - Linear interpolation draws a flat line
# - Random Forest predicts warm during day, cool at night
#   (because it learned that high radiation = high temp)

# In my experience, Random Forest reduces RMSE by 30-50% 
# compared to linear interpolation for day-long gaps.

# But for gaps under 3-6 hours, the methods are often similar
# because temperature hasn't changed much anyway.
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

---

# **Wrapping Up**

We've covered a lot of ground here. Let me leave you with the key takeaways:

**Simple regression** is your starting point. It's easy to understand, easy to explain, and often good enough for straightforward questions.

**Multiple regression** lets you account for multiple drivers at once. The coefficients tell you the effect of each variable while controlling for the others.

**Random Forests** can capture complex patterns that regression misses. They're particularly good when you don't know the shape of the relationships in advance.

**For gap-filling**, match your method to your gap length. Simple interpolation for short gaps, model-based methods for longer ones.

**Most importantly:** always plot your data, check your assumptions, and validate on independent test data. No amount of fancy statistics can fix bad data or inappropriate models.

Good luck with your analyses!

## Where to Go From Here

If you want to dig deeper:
- **Generalized Additive Models (GAMs)** let you fit smooth curves instead of straight lines
- **Mixed-effects models** handle hierarchical data (e.g., measurements nested within sites)
- **Gradient Boosting (XGBoost)** often outperforms Random Forests for prediction
- **Time series methods** (ARIMA, etc.) are specifically designed for temporal data

But honestly, you can get surprisingly far with just regression and Random Forests. Master these first before moving on to fancier tools.
